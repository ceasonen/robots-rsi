"""Actual demonstration sensors, synthetic returns: engineering only.

No physical episode or trained research actor is created. This validates zero
LoRA parity, cached-history behavior density and actual source-observation
gradients using the author's released initialization on explicit CPU float32.
"""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"rsi_group_meeting"))


def digest_parameters(named):
    result=hashlib.sha256()
    for name,value in named:
        result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    import h5py
    import numpy as np
    import torch
    from dmpel_native_backend_v6 import NativeDMPELSource90
    from dmpel_actor_runtime_v7 import ActorRuntime

    output=ROOT/"reports/dmpel-actor-runtime-v7-cpu-check.json"
    assert not output.exists()
    path=ROOT/"data/datasets/official_libero90/KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_demo.hdf5"
    language="put the butter at the back in the top drawer of the cabinet and close it"
    with h5py.File(path,"r") as handle:
        demo=handle["data/demo_0"]
        def sensors(index):
            return {"agentview_image":demo["obs/agentview_rgb"][index],"robot0_eye_in_hand_image":demo["obs/eye_in_hand_rgb"][index],
                    "robot0_joint_pos":demo["obs/joint_states"][index],"robot0_gripper_qpos":demo["obs/gripper_states"][index]}
        observations=[sensors(0),sensors(1)]
    native=NativeDMPELSource90("cpu",threads=2)
    before=digest_parameters(native.policy.named_parameters())
    native.reset(97931)
    original=[native.query(raw,language) for raw in observations]
    runtime=ActorRuntime(native,77210)
    native.reset(97931)
    current=[runtime.query(raw,language)[0] for raw in observations]
    parity=float(np.max(np.abs(np.asarray(original)-np.asarray(current))))
    assert parity<=1e-5
    episodes=[]
    for index in range(4):
        native.reset(97940+index)
        actions,features,logps=[],[],[]
        for raw in observations:
            action,logp,feature,_=runtime.query(raw,language)
            actions.append(action);logps.append(logp);features.append(feature)
        episodes.append({"actions":np.asarray(actions,dtype=np.float32),"spatial_features":np.asarray(features,dtype=np.float32),
                         "behavior_log_probabilities":np.asarray(logps),"success":bool(index%2==0)})
    density=runtime.check_behavior_density(episodes)
    optimizer=torch.optim.AdamW(runtime.parameters,lr=1e-4,weight_decay=0)
    zero=[episode|{"success":True} for episode in episodes]
    zero_result=runtime.update(zero,optimizer,mode="rl")
    assert zero_result["optimizer_steps"]==0
    trained=runtime.update(episodes,optimizer,mode="rl")
    assert trained["optimizer_steps"]==2
    changes=[]
    with torch.no_grad():
        for episode in episodes:
            for index,action in enumerate(episode["actions"]):
                distribution=runtime.distribution(runtime.history(episode["spatial_features"],index))
                changed=distribution.log_prob(torch.from_numpy(action).unsqueeze(0))
                changes.append(abs(float(changed)-float(episode["behavior_log_probabilities"][index])))
    assert max(changes)>0 and before==digest_parameters(runtime.frozen)
    runtime.load(runtime.initial)
    bc_optimizer=torch.optim.AdamW(runtime.parameters,lr=1e-4,weight_decay=0)
    bc=runtime.update(episodes,bc_optimizer,mode="successful_bc")
    assert bc["optimizer_steps"]==2 and before==digest_parameters(runtime.frozen)
    report={"checked_utc":datetime.now(UTC).isoformat(),"status":"passed_actual_source_sensor_cpu_actor_runtime_engineering",
        "zero_LoRA_action_difference":parity,"behavior_density_check":density,"uninformative_initial_group":zero_result,
        "RL_synthetic_return_update":trained,"successful_BC_synthetic_return_update":bc,
        "maximum_log_probability_change_after_engineering_RL_steps":max(changes),
        "frozen_pretrained_parameters_sha256_before_after":before,"metadata":native.metadata,
        "observations":"First two actual frames of original source task3 demo0; OpenGL128 RGB, original joint7/gripper2",
        "rewards":"Synthetic [1,0,1,0] returns only; no recorded or newly evaluated task outcome is used",
        "physical_trials":0,"research_actor_saved":False,"research_training_updates":0,"GPU_path_validated":False,
        "implementation_sha256":hashlib.sha256((ROOT/"rsi_group_meeting/dmpel_actor_runtime_v7.py").read_bytes()).hexdigest()}
    output.write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()

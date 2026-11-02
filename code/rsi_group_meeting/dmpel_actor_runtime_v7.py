"""Low-rank GMM actor with source-only on-policy update operations.

No collection or experiment launch occurs here. GPU actor collection and
re-scoring use identical float16 temporal operators and canonical float32
mixture densities. CPU engineering checks use an explicit float32 port.
"""
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from dmpel_motor_core_v6 import add_temporal_lora, distribution_from_frozen_history, leave_one_out_returns
from dmpel_onpolicy_objective_v7 import step_objective
from dmpel_policy_ports_v6 import canonical_query


class ActorRuntime:
    def __init__(self, native, seed):
        self.native = native
        self.policy = native.policy
        self.frozen = list(self.policy.named_parameters())
        self.parameters, self.names = add_temporal_lora(self.policy, seed)
        self.policy.eval()
        self.layers = [module for module in self.policy.modules() if hasattr(module,"lora_a") and hasattr(module,"lora_b")]
        assert len(self.layers)==6 and sum(p.numel() for p in self.parameters)==73728
        assert all(p.dtype==torch.float32 for p in self.parameters)
        self.initial = self.state()

    def context(self):
        return torch.amp.autocast("cuda",dtype=torch.float16) if self.native.device=="cuda" else nullcontext()

    def state(self):
        return {name:value.detach().cpu().clone() for name,value in self.policy.state_dict().items() if ".lora_" in name}

    def load(self, state):
        assert set(state)==set(self.initial)
        assert all(value.dtype==torch.float32 and torch.isfinite(value).all() for value in state.values())
        result=self.policy.load_state_dict(state,strict=False)
        assert not result.unexpected_keys and not set(state).intersection(result.missing_keys)
        self.policy.eval()

    @contextmanager
    def reference(self):
        scales=[module.scale for module in self.layers]
        try:
            for module in self.layers:
                module.scale=0.
            yield
        finally:
            for module,scale in zip(self.layers,scales):
                module.scale=scale

    def query(self,raw,language):
        return canonical_query(self.native,raw,language)

    def history(self,features,index):
        assert features.ndim==3 and features.shape[1:]==(4,768)
        # The author's FiLM fusion promotes the frozen spatial output to F32
        # even under CUDA F16 autocast. Preserve those actual values exactly.
        assert features.dtype==np.float32
        return torch.from_numpy(features[max(0,index-9):index+1].copy()).unsqueeze(0).to(self.native.device)

    def distribution(self,history):
        with self.context():
            return distribution_from_frozen_history(self.policy,history)

    def check_behavior_density(self, episodes, tolerance=1e-5):
        """Check every stored action before any optimization, with grad enabled."""
        maximum=0.;steps=0
        for episode in episodes:
            assert len(episode["spatial_features"])==len(episode["actions"])==len(episode["behavior_log_probabilities"])
            for index,action in enumerate(episode["actions"]):
                history=self.history(episode["spatial_features"],index)
                distribution=self.distribution(history)
                action=torch.as_tensor(action,device=self.native.device,dtype=torch.float32).unsqueeze(0)
                current=distribution.log_prob(action)
                difference=abs(float(current.detach().cpu())-float(episode["behavior_log_probabilities"][index]))
                maximum=max(maximum,difference);steps+=1
                assert difference<=tolerance, f"On-policy density mismatch {difference}; do not train"
        return {"maximum_absolute_log_probability_difference":maximum,"actions_checked":steps,
                "tolerance_predeclared":tolerance,"recomputation_has_grad_enabled":True}

    def update(self,episodes,optimizer,*,mode,epochs=2,clip=.2,kl_coefficient=.001,max_grad_norm=1.):
        """A four-trajectory group with official terminal binary rewards.

        RL uses an independent leave-one-out return baseline. BC is explicitly
        successful-trajectory cloning on the same RL-collected data; it is not
        claimed to collect experience from its own current policy.
        """
        assert mode in ["rl","successful_bc"] and len(episodes)==4 and epochs>0
        assert all(episode["success"] in [0,1,False,True] for episode in episodes)
        rewards=[float(episode["success"]) for episode in episodes]
        advantages=leave_one_out_returns(rewards).to(self.native.device)
        informative=bool(torch.any(advantages!=0))
        selected=episodes if mode=="rl" else [episode for episode in episodes if episode["success"]]
        denominator=len(episodes) if mode=="rl" else len(selected)
        if not selected or (mode=="rl" and not informative and all(torch.count_nonzero(layer.lora_b)==0 for layer in self.layers)):
            return {"mode":mode,"optimizer_steps":0,"official_success_rewards":rewards,
                    "informative_group":informative,"reason":"No nonzero task-reward gradient and zero reference penalty"}
        metrics=[]
        self.policy.eval()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            total_loss=0.;total_kl=0.;density_drift=0.;controls=0
            for episode_index,episode in enumerate(episodes):
                if mode=="successful_bc" and not episode["success"]:
                    continue
                for index,action in enumerate(episode["actions"]):
                    history=self.history(episode["spatial_features"],index)
                    action=torch.as_tensor(action,device=self.native.device,dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad(), self.reference():
                        reference=self.distribution(history)
                    current=self.distribution(history)
                    logp=current.log_prob(action)
                    if mode=="rl":
                        old=torch.tensor([float(episode["behavior_log_probabilities"][index])],device=self.native.device)
                        loss,parts=step_objective(logp,old,advantages[episode_index:episode_index+1],current,reference,
                                                clip=clip,kl_coefficient=kl_coefficient)
                        density_drift=max(density_drift,float((logp.detach()-old).abs().max().cpu()))
                    else:
                        from dmpel_onpolicy_objective_v7 import gmm_joint_kl
                        kl=gmm_joint_kl(current,reference)
                        loss=-logp+kl_coefficient*kl
                        parts={"joint_kl_upper_bound":kl}
                    (loss.sum()/denominator).backward()
                    total_loss+=float(loss.detach().sum().cpu())/denominator
                    total_kl+=float(parts["joint_kl_upper_bound"].detach().sum().cpu())/denominator
                    controls+=1
            assert all(parameter.grad is None for _,parameter in self.frozen)
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in self.parameters)
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,max_grad_norm)
            assert torch.isfinite(norm)
            optimizer.step()
            assert all(torch.isfinite(parameter).all() for parameter in self.parameters)
            metrics.append({"epoch":epoch,"summed_step_objective_per_trajectory":total_loss,
                "summed_joint_KL_per_trajectory":total_kl,"gradient_norm_before_clip":float(norm.cpu()),
                "maximum_behavior_log_probability_drift":density_drift,"controls_in_loss":controls})
        optimizer.zero_grad(set_to_none=True)
        return {"mode":mode,"optimizer_steps":epochs,"official_success_rewards":rewards,
            "informative_group":informative,"epochs":metrics,"per_trajectory_length_normalization":False,
            "reference_KL_kind":"exact_auxiliary_component_action_joint_KL_upper_bound"}

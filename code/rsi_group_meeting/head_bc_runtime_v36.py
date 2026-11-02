"""Standard head-input LoRA, with explicitly preserved native output dtype."""
from contextlib import contextmanager,nullcontext
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dmpel_motor_core_v6 import distribution_from_frozen_history
from dmpel_policy_ports_v6 import canonical_query

class HeadInputLowRankLinear(nn.Module):
    def __init__(self,base,rank=72,alpha=144):
        super().__init__();assert isinstance(base,nn.Linear) and (base.in_features,base.out_features)==(768,256)
        self.base=base.requires_grad_(False);self.rank=rank;self.alpha=alpha;self.scale=alpha/rank
        self.lora_a=nn.Parameter(torch.empty(rank,768,device=base.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(256,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
    def forward(self,value):
        original=self.base(value)
        residual=F.linear(F.linear(value,self.lora_a),self.lora_b)*self.scale
        # An F32 residual must never promote the author's F16 head activations.
        return original+residual.to(dtype=original.dtype)

class HeadOnlyRuntime:
    def __init__(self,native,seed=77210):
        self.native=native;self.policy=native.policy
        self.policy.eval().requires_grad_(False);self.frozen=list(self.policy.named_parameters())
        assert not any('.lora_' in name for name,_ in self.frozen),'Use an original author model, not a temporal-adapter layout.'
        torch.manual_seed(seed)
        self.policy.policy_head.share[0]=HeadInputLowRankLinear(self.policy.policy_head.share[0],72,144)
        self.layer=self.policy.policy_head.share[0]
        self.parameters=[self.layer.lora_a,self.layer.lora_b]
        assert sum(v.numel() for v in self.parameters)==73728
        self.names=['policy_head.share.0.lora_a','policy_head.share.0.lora_b']
        assert [name for name,v in self.policy.named_parameters() if v.requires_grad]==self.names
        self.policy.policy_head.low_eval_noise=False
    def context(self):return torch.amp.autocast('cuda',dtype=torch.float16) if self.native.device=='cuda' else nullcontext()
    def state(self):return {name:value.detach().cpu().clone() for name,value in self.policy.state_dict().items() if name in self.names}
    def load(self,state):
        assert set(state)==set(self.names)
        for name,shape in zip(self.names,[(72,768),(256,72)]):
            assert state[name].dtype==torch.float32 and tuple(state[name].shape)==shape and torch.isfinite(state[name]).all()
        result=self.policy.load_state_dict(state,strict=False);assert not result.unexpected_keys and not set(state).intersection(result.missing_keys)
        self.policy.eval();self.policy.policy_head.low_eval_noise=False
    @contextmanager
    def reference(self):
        scale=self.layer.scale
        try:self.layer.scale=0.;yield
        finally:self.layer.scale=scale
    def history(self,features,index):
        assert features.ndim==3 and features.shape[1:]==(4,768) and features.dtype==np.float32
        return torch.from_numpy(features[max(0,index-9):index+1].copy()).unsqueeze(0).to(self.native.device)
    def distribution(self,history):
        with self.context():return distribution_from_frozen_history(self.policy,history)
    def query(self,raw,language):return canonical_query(self.native,raw,language)

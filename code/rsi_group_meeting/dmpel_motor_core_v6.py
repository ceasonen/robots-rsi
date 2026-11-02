"""Prospective GMM-policy low-rank PPO primitives, not a trained research actor.

The GMM admits an analytic action density. Canonical float32 mixture sampling
and density evaluation are explicit policy-port choices; the author's CUDA
half-precision sampling path is not claimed identical. All comparison arms
must share this port and pass physical source capability checks first.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalLowRankLinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, value):
        return self.base(value) + F.linear(F.linear(value, self.lora_a), self.lora_b) * self.scale


def add_temporal_lora(policy, seed):
    policy.eval().requires_grad_(False)
    torch.manual_seed(seed)
    names = []
    for name, module in list(policy.named_modules()):
        if name.startswith("temporal_transformer.") and name.endswith(".qkv") and isinstance(module, nn.Linear):
            parent_name, leaf = name.rsplit(".", 1)
            setattr(policy.get_submodule(parent_name), leaf, TemporalLowRankLinear(module))
            names.append(name)
    assert len(names) == 6
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    assert all("lora_" in name for name, parameter in policy.named_parameters() if parameter.requires_grad)
    return parameters, names


def canonical_gmm(distribution):
    assert isinstance(distribution, torch.distributions.MixtureSameFamily)
    assert isinstance(distribution.component_distribution, torch.distributions.Independent)
    normal = distribution.component_distribution.base_dist
    assert isinstance(normal, torch.distributions.Normal)
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=distribution.mixture_distribution.logits.float()),
        torch.distributions.Independent(torch.distributions.Normal(normal.loc.float(), normal.scale.float()), 1))


def distribution_from_frozen_history(policy, history):
    assert history.ndim == 4 and history.shape[-1] == 768 and 1 <= history.shape[1] <= 10
    assert not history.requires_grad, "Cache only frozen spatial features, never actor-dependent temporal features"
    hidden = policy.temporal_encode(history)
    return canonical_gmm(policy.policy_head(hidden[:, -1]))


def leave_one_out_returns(returns):
    returns = torch.as_tensor(returns, dtype=torch.float32)
    assert returns.ndim == 1 and len(returns) >= 2 and torch.isfinite(returns).all()
    return returns - (returns.sum() - returns) / (len(returns) - 1)


def clipped_policy_surrogate(log_probability, old_log_probability, advantages, clip=.2):
    assert log_probability.shape == old_log_probability.shape == advantages.shape
    assert 0 < clip < 1
    log_ratio = log_probability - old_log_probability
    assert torch.isfinite(log_ratio).all() and log_ratio.detach().abs().max() < 60
    ratio = log_ratio.exp()
    value = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    return -value

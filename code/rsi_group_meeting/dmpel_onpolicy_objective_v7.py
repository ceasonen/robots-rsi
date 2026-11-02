"""Analytic GMM-path trust penalty and trajectory-level policy-gradient terms.

The mixture component is an auxiliary variable with the same fixed component
identity in current and frozen policies. Its exact joint KL upper-bounds the
marginal action KL; it is not claimed to equal the intractable mixture KL.
"""
import torch


def gmm_joint_kl(current, reference):
    """KL(q(z,a)||p0(z,a)) in float32, retaining gradients through q."""
    assert isinstance(current, torch.distributions.MixtureSameFamily)
    assert isinstance(reference, torch.distributions.MixtureSameFamily)
    q = current.component_distribution.base_dist
    p = reference.component_distribution.base_dist
    assert isinstance(q, torch.distributions.Normal) and isinstance(p, torch.distributions.Normal)
    assert q.loc.shape == p.loc.shape and q.loc.shape[-2:] == (5, 7)
    q_mean, q_scale = q.loc.float(), q.scale.float()
    p_mean, p_scale = p.loc.detach().float(), p.scale.detach().float()
    q_log = current.mixture_distribution.logits.float()
    p_log = reference.mixture_distribution.logits.detach().float()
    component_kl = (p_scale.log() - q_scale.log()
        + (q_scale.square() + (q_mean-p_mean).square()) / (2*p_scale.square()) - .5).sum(-1)
    return (q_log.exp() * (q_log-p_log+component_kl)).sum(-1)


def step_objective(current_log_probability, old_log_probability, advantage,
                   current_distribution, reference_distribution, *, clip=.2, kl_coefficient=.001):
    """One action's contribution, later summed within each trajectory.

    The caller divides by the number of trajectories, not each realized
    trajectory length. At the first unchanged-policy pass, the unclipped term
    therefore has the episodic score-function scaling with an independent
    leave-one-out return baseline. PPO clipping is explicitly a surrogate.
    """
    from dmpel_motor_core_v6 import clipped_policy_surrogate
    assert kl_coefficient >= 0
    policy = clipped_policy_surrogate(current_log_probability, old_log_probability, advantage, clip)
    trust = gmm_joint_kl(current_distribution, reference_distribution)
    assert policy.shape == trust.shape
    return policy + kl_coefficient*trust, {"policy": policy, "joint_kl_upper_bound": trust}

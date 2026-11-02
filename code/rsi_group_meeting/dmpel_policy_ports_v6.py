"""Explicit GMM sampling port and cache, without changing the native backend."""

from contextlib import nullcontext

from dmpel_motor_core_v6 import canonical_gmm


def canonical_query(native, raw, language):
    torch, np = native.torch, native.np
    for name in ["agentview_image", "robot0_eye_in_hand_image"]:
        assert np.asarray(raw[name]).shape == (128, 128, 3) and np.asarray(raw[name]).dtype == np.uint8
    assert np.asarray(raw["robot0_joint_pos"]).shape == (7,)
    assert np.asarray(raw["robot0_gripper_qpos"]).shape == (2,)
    if language not in native._tokens:
        tokens = native.tokenizer(text=[language], add_special_tokens=True, max_length=native.cfg.data.max_word_len,
                                  padding="max_length", return_attention_mask=True, return_tensors="pt")
        native._tokens[language] = {key: tokens[key][0] for key in ["attention_mask", "input_ids"]}
    data = native.preprocess([raw], native._tokens[language], native.cfg)
    context = torch.amp.autocast("cuda", dtype=torch.float16) if native.device == "cuda" else nullcontext()
    with torch.no_grad():
        with context:
            data = native.policy.preprocess_input(data, train_mode=False)
            spatial = native.policy.spatial_encode(data)
            native.policy.latent_queue.append(spatial)
            if len(native.policy.latent_queue) > native.policy.max_seq_len:
                native.policy.latent_queue.pop(0)
            hidden = native.policy.temporal_encode(torch.cat(native.policy.latent_queue, dim=1))
            original = native.policy.policy_head(hidden[:, -1])
        distribution = canonical_gmm(original)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
    assert action.shape == (1, 7) and torch.isfinite(action).all() and torch.isfinite(log_probability).all()
    return action[0].cpu().numpy(), float(log_probability[0].cpu()), spatial[0, 0].cpu().numpy(), distribution

"""Real public SmolVLA inference on pinned LIBERO / official LIBERO-Pro assets.

No privileged object poses are passed to the policy. This is a runtime-patch
pilot, not an implementation of ASPIRE, RHO, RPG, or VLA RL+LoRA training.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rsi_group_meeting/v4"
REPO = ROOT / "external/LIBERO-PRO"
MODEL = ROOT / "data/models/smolvla_libero"
VLM = ROOT / "data/models/smolvlm2_config"
PRO = ROOT / "data/libero_pro"
PACK = REPO / "libero/libero"
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(REPO))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(path)


def language(path):
    match = re.search(r"\(:language\s+(.*?)\)", path.read_text(encoding="utf8"), re.DOTALL)
    if match is None:
        raise ValueError(f"Missing authoritative BDDL language: {path}")
    return " ".join(match.group(1).split()).strip('"')


def make_plan():
    """Freeze task names and state indices before observing any rollout outcome."""
    import torch
    from libero.libero import benchmark
    from paired_observation_v4 import CONTRACT

    asset = json.loads((OUT / "asset_receipt.json").read_text(encoding="utf8"))
    lfs = next(x["lfs"] for x in asset["model"]["siblings"] if x["rfilename"] == "model.safetensors")
    expected = lfs["sha256"]
    assert MODEL.joinpath("model.safetensors").stat().st_size == lfs["size"], "Weights download incomplete"
    assert digest(MODEL / "model.safetensors") == expected, "Weights checksum mismatch"
    settings = []
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        tasks = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        for task_id in [0, 1]:
            task = tasks.get_task(task_id)
            for perturb in ["original", "swap"]:
                bddl_root = PACK if perturb == "original" else PRO
                folder = suite if perturb == "original" else suite + "_swap"
                bddl = bddl_root / "bddl_files" / folder / task.bddl_file
                init = bddl_root / "init_files" / folder / task.init_states_file
                assert bddl.exists() and init.exists(), (bddl, init)
                states = torch.load(init, weights_only=False)
                assert len(states) >= 4, f"Insufficient unique saved states for disjoint split: {init}"
                settings.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "task_name": task.name,
                        "perturbation": perturb,
                        "language": task.language,
                        "bddl_language": language(bddl),
                        "bddl": str(bddl.relative_to(ROOT)).replace("\\", "/"),
                        "bddl_sha256": digest(bddl),
                        "init_file": str(init.relative_to(ROOT)).replace("\\", "/"),
                        "init_sha256": digest(init),
                        "n_saved_states": len(states),
                        "max_steps": {
                            "libero_spatial": 280,
                            "libero_object": 280,
                            "libero_goal": 300,
                            "libero_10": 520,
                        }[suite],
                    }
                )
    plan = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_formal_rollouts_of_this_revision",
        "protocol_version": 2,
        "initial_observation_pairing": CONTRACT,
        "asset_receipt_sha256": digest(OUT / "asset_receipt.json"),
        "weights_sha256": expected,
        "settings": settings,
        "source_state_indices": [0, 1],
        "heldout_state_indices": [2, 3],
        "pilot_integration_state_index": 4,
        "policy_seed": 20261019,
        "settling_steps": 10,
        "control_hz": 20,
        "rgb_resolution": [256, 256],
        "image_rotation": "180 degrees, official LeRobot LiberoProcessorStep",
        "observations": "agentview RGB, wrist RGB, eef position/axis-angle, gripper joint positions, official benchmark task.language (as in LeRobot); Swap preserves the instruction",
        "success": "unchanged official env.check_success; any success within the official suite horizon",
        "precision": "Checkpoint parameter dtypes: 149570176 float32 and 455364000 bfloat16 parameters; autocast disabled. BF16 autocast tested on integration state but rejected because it did not reliably accelerate inference.",
        "batch_size": 2,
        "batching": "Fixed rows, independent simulators. Source: consecutive 2 original task/state pairs. Heldout: consecutive 2 setting/state pairs. Completed rows stay as padding. One shared policy RNG seed per batch, reset for each candidate. Four-environment integration succeeded but used about 5.5/6 GB total VRAM; reduced to 2 for graphics-memory headroom before any formal rollout completed.",
        "checkpoint_default_action_steps": 1,
        "evaluation_action_steps": 10,
        "inference_configuration_boundary": "Official select_action chunk interface, evaluation action_steps=10 (0.5 simulated seconds per prediction). This differs from the checkpoint's saved action_steps=1. Same configuration for base and translation-scale repairs; shorter_chunk explicitly changes to 5. Do not call these exact published default-policy scores.",
        "variants": {
            "base": {"translation_scale": 1.0, "action_steps": 10},
            "slower_translation": {"translation_scale": 0.8, "action_steps": 10},
            "faster_translation": {"translation_scale": 1.2, "action_steps": 10},
            "shorter_chunk": {"translation_scale": 1.0, "action_steps": 5},
        },
        "scope": "8 tasks x original/swap. Small prespecified pilot, not complete benchmark, not paper method reproduction. Runtime patch candidates are not LoRA adapters.",
        "interpretation": "Differences across suites are observational; tasks differ, so this alone cannot identify a causal effect of complexity.",
    }
    path = OUT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        plan["created_utc"] = old["created_utc"]
        assert old == plan, "Frozen protocol differs; create a new experiment rather than overwrite"
    else:
        write_json(path, plan)
    print("Frozen settings", len(settings), "weights verified", flush=True)
    return plan


def load_policy():
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.env_processor import LiberoProcessorStep

    torch.set_num_threads(8)
    config = PreTrainedConfig.from_pretrained(str(MODEL))
    assert isinstance(config, SmolVLAConfig)
    config.load_vlm_weights = False  # All weights strictly loaded from the complete LIBERO checkpoint below.
    config.vlm_model_name = str(VLM)
    policy = SmolVLAPolicy.from_pretrained(str(MODEL), config=config, strict=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(MODEL),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(VLM)}},
    )
    return policy, pre, post, LiberoProcessorStep()


def observation_batch(raw, instruction, env_processor):
    import torch

    # Exact official LeRobot env adapter convention, including quaternion conversion.
    batch = {"task": [instruction]}
    for raw_key, policy_key in [("agentview_image", "image"), ("robot0_eye_in_hand_image", "image2")]:
        batch[f"observation.images.{policy_key}"] = (
            torch.from_numpy(raw[raw_key].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        )
    batch["observation.robot_state"] = {
        "eef": {
            "pos": torch.as_tensor(raw["robot0_eef_pos"]).unsqueeze(0),
            "quat": torch.as_tensor(raw["robot0_eef_quat"]).unsqueeze(0),
        },
        "gripper": {"qpos": torch.as_tensor(raw["robot0_gripper_qpos"]).unsqueeze(0)},
    }
    return env_processor.observation(batch)


def rollout(setting, state_index, variant, plan, model, max_steps=None, integration=False):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    policy, pre, post, env_processor = model
    variant_config = plan["variants"][variant]
    key = f"{setting['suite']}_task{setting['task_id']}_{setting['perturbation']}_init{state_index}_{variant}"
    folder = OUT / ("integration" if integration else "results")
    receipt = folder / (key + ".json")
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf8"))
        assert previous["protocol_sha256"] == digest(OUT / "protocol.json")
        print("Reusing", key, previous["success"], flush=True)
        return previous
    for name, expected in [("bddl", "bddl_sha256"), ("init_file", "init_sha256")]:
        assert digest(ROOT / setting[name]) == setting[expected]
    states = torch.load(ROOT / setting["init_file"], weights_only=False)
    assert state_index < len(states), "Never modulo-wrap initial state indices"
    torch.manual_seed(plan["policy_seed"])
    np.random.seed(plan["policy_seed"])
    torch.cuda.reset_peak_memory_stats()
    policy.config.n_action_steps = variant_config["action_steps"]
    policy.reset()
    env = OffScreenRenderEnv(
        bddl_file_name=str(ROOT / setting["bddl"]),
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        hard_reset=False,
    )
    env.seed(plan["policy_seed"])
    video_path = OUT / "videos" / (key + ".mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    actions, rewards, decision_times, robot_states = [], [], [], []
    first_features = []
    initial_features = None
    first_control_utc = None

    # Frozen vision representation from the first policy observation only; no object state.
    def capture(_module, _inputs, output):
        if len(first_features) < 2:
            hidden = output.last_hidden_state
            first_features.append(hidden.detach().float().mean(dim=1).cpu().numpy())

    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    hook = vision.register_forward_hook(capture)
    success = False
    try:
        raw = env.reset()
        raw = env.set_init_state(states[state_index])
        for _ in range(plan["settling_steps"]):
            raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        initial_sim_hash = hashlib.sha256(np.array(env.get_sim_state()).tobytes()).hexdigest()
        initial_state = (
            observation_batch(raw, setting["language"], env_processor)["observation.state"].numpy().tolist()
        )
        limit = setting["max_steps"] if max_steps is None else max_steps
        with imageio.get_writer(
            video_path, fps=20, codec="libx264", ffmpeg_params=["-crf", "18", "-movflags", "+faststart"]
        ) as writer:
            for step in range(limit):
                writer.append_data(
                    np.concatenate(
                        [raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"][::-1, ::-1]],
                        axis=1,
                    )
                )
                tick = time.perf_counter()
                batch = pre(observation_batch(raw, setting["language"], env_processor))
                with torch.inference_mode():
                    action = post(policy.select_action(batch)).detach().cpu().numpy().reshape(-1)
                    if step == 0:
                        tokens = batch["observation.language.tokens"]
                        mask = batch["observation.language.attention_mask"].float().unsqueeze(-1)
                        text_hidden = policy.model.vlm_with_expert.embed_language_tokens(tokens)
                        text_mean = (text_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                        initial_features = np.concatenate(
                            [
                                *[x.reshape(-1) for x in first_features],
                                text_mean.cpu().numpy().reshape(-1),
                                np.asarray(initial_state).reshape(-1),
                            ]
                        ).astype(np.float32)
                        hook.remove()
                        if not integration and state_index in plan["heldout_state_indices"]:
                            from selection_v4 import freeze_choice

                            freeze_choice(setting, state_index, initial_features, initial_sim_hash)
                action[:3] *= variant_config["translation_scale"]
                action = np.clip(action, -1.0, 1.0)
                decision_times.append(time.perf_counter() - tick)
                if step == 0:
                    first_control_utc = datetime.now(UTC).isoformat()
                raw, reward, done, _info = env.step(action)
                actions.append(action.copy())
                rewards.append(float(reward))
                robot_states.append(
                    np.concatenate(
                        [raw["robot0_eef_pos"], raw["robot0_eef_quat"], raw["robot0_gripper_qpos"]]
                    )
                )
                success = bool(env.check_success())
                if step % 50 == 0:
                    print(
                        key,
                        step,
                        "success",
                        success,
                        "seconds",
                        round(time.perf_counter() - start, 1),
                        flush=True,
                    )
                if success or done:
                    break
            writer.append_data(
                np.concatenate(
                    [raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"][::-1, ::-1]], axis=1
                )
            )
        folder.mkdir(parents=True, exist_ok=True)
        trace_path = folder / (key + ".npz")
        np.savez_compressed(
            trace_path,
            actions=actions,
            rewards=rewards,
            robot_states=robot_states,
            decision_times=decision_times,
            initial_features=initial_features,
            initial_vision_features=np.concatenate(first_features, axis=-1)
            if first_features
            else np.empty(0),
        )
        result = {
            "key": key,
            "suite": setting["suite"],
            "task_id": setting["task_id"],
            "task_name": setting["task_name"],
            "perturbation": setting["perturbation"],
            "state_index": state_index,
            "variant": variant,
            "success": success,
            "steps": len(actions),
            "horizon": limit,
            "wall_seconds": time.perf_counter() - start,
            "simulated_seconds": (len(actions) + plan["settling_steps"]) / plan["control_hz"],
            "initial_robot_state": initial_state,
            "initial_sim_sha256": initial_sim_hash,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "mean_action_latency_seconds": float(np.mean(decision_times)),
            "protocol_sha256": digest(OUT / "protocol.json"),
            "video": str(video_path.relative_to(OUT)).replace("\\", "/"),
            "trace_sha256": digest(trace_path),
            "integration_only": integration,
            "first_control_utc": first_control_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "runner_code_sha256": digest(Path(__file__)),
        }
        write_json(receipt, result)
        print("COMPLETED", key, "success", success, "wall", round(result["wall_seconds"], 1), flush=True)
        return result
    finally:
        hook.remove()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["plan", "integration", "base", "source", "heldout", "all"])
    parser.add_argument("--limit-settings", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    plan = make_plan()
    if args.stage == "plan":
        return
    if args.stage not in ["integration", "plan"]:
        raise RuntimeError(
            "This protocol requires batch_benchmark_v4.py; do not mix scalar and batched RNG protocols"
        )
    if args.stage == "all":
        from selection_v4 import freeze_protocol

        freeze_protocol()
    model = load_policy()
    settings = plan["settings"][: args.limit_settings]
    if args.stage == "integration":
        rollout(
            settings[0],
            plan["pilot_integration_state_index"],
            "base",
            plan,
            model,
            args.max_steps or 30,
            True,
        )
        return
    assert args.max_steps is None, "A shortened rollout cannot be recorded as a benchmark score"
    if args.stage == "all":
        from selection_v4 import train

        rollout(settings[0], plan["pilot_integration_state_index"], "base", plan, model, 30, True)
        for setting in settings:
            if setting["perturbation"] == "original":
                for state_index in plan["source_state_indices"]:
                    for variant in plan["variants"]:
                        rollout(setting, state_index, variant, plan, model)
        if not (OUT / "models/freeze_receipt.json").exists():
            train()
        for setting in settings:
            for state_index in plan["heldout_state_indices"]:
                for variant in plan["variants"]:
                    rollout(setting, state_index, variant, plan, model)
        return
    for setting in settings:
        if args.stage == "source" and setting["perturbation"] != "original":
            continue
        indices = plan["source_state_indices"] if args.stage == "source" else plan["heldout_state_indices"]
        variants = ["base"] if args.stage == "base" else list(plan["variants"])
        for state_index in indices:
            for variant in variants:
                rollout(setting, state_index, variant, plan, model)


if __name__ == "__main__":
    main()

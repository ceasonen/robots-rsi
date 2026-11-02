"""Prospective physical source screen of an author source90 initialization.

Two explicit sampling precisions; same frozen weights and24 physical initial
states. This is not RL improvement, heldout transfer, or published-score
reproduction. GPU remains exclusive and no target/Long trajectory is run.
"""

import argparse
import gc
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "rsi_group_meeting/v6/dmpel_source_probe"
PRIMARY = ROOT / "rsi_group_meeting/v4/xvla"
SOURCE = ROOT / "rsi_group_meeting/v4/source90_candidate"
SENSORS = ["agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".working.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    temporary.replace(path)


def bindings():
    names = ["rsi_group_meeting/evaluate_dmpel_source_probe_v6.py", "rsi_group_meeting/dmpel_probe_client_v6.py",
             "scripts/dmpel_probe_worker_v6.py", "scripts/dmpel_native_backend_v6.py",
             "rsi_group_meeting/dmpel_policy_ports_v6.py", "rsi_group_meeting/dmpel_motor_core_v6.py",
             "rsi_group_meeting/xvla_benchmark_v4.py", "rsi_group_meeting/paired_observation_v4.py",
             "rsi_group_meeting/benchmark_v4.py", "rsi_group_meeting/probe_source_reference_v4.py",
             "rsi_group_meeting/local_run_lock_v4.py"]
    return {name: sha(ROOT / name) for name in names}


def register():
    path = WORK / "registration.json"
    if path.exists():
        plan = read(path)
        assert plan["implementation_sha256"] == bindings()
        return plan
    assert not list((WORK / "results").glob("*.json"))
    assert not list((PRIMARY / "results/heldout").glob("*.json"))
    engineering = read(ROOT / "reports/dmpel-source90-cpu-engineering-check-v6.json")
    assert engineering["status"] == "passed_strict_author_initialization_and_synthetic_cpu_schema_check"
    assert engineering["metadata"]["backend_sha256"] == sha(ROOT / "scripts/dmpel_native_backend_v6.py")
    core = read(ROOT / "reports/dmpel-motor-core-v6-cpu-gradient-check.json")
    assert core["status"] == "passed_actual_author_weight_cpu_zero_lora_and_analytic_density_gradient_check"
    assert core["implementation_sha256"] == sha(ROOT / "rsi_group_meeting/dmpel_motor_core_v6.py")
    original = read(SOURCE / "registration.json")
    plan = {"registered_utc": datetime.now(UTC).isoformat(),
        "scope": "Source-only initialization/sampling-precision capability screen, no actor update or heldout evidence",
        "source_task_ids": original["source_task_ids"], "states": [0, 1], "policy_seed": 20261020,
        "variants": ["native_half", "canonical_float32"], "expected_episodes": 48,
        "physical_initial_states": 24, "semantic_tasks": 12, "horizon": 600,
        "shared_budget_sensitivities": [400, 520], "source_protocol_sha256": sha(PRIMARY / "protocol.json"),
        "initialization_asset_receipt_sha256": sha(ROOT / "data/models/dmpel_author_source90/asset_receipt.json"),
        "clip_asset_receipt_sha256": sha(ROOT / "data/models/dmpel_clip_config_tokenizer/asset_receipt.json"),
        "cpu_engineering_check_sha256": sha(ROOT / "reports/dmpel-source90-cpu-engineering-check-v6.json"),
        "cpu_gradient_check_sha256": sha(ROOT / "reports/dmpel-motor-core-v6-cpu-gradient-check.json"),
        "implementation_sha256": bindings(),
        "pixel_contract": "Native model uses raw128 RGB and joint7/gripper2 without180-degree rotation; a separate128 canonical sensor cache is used",
        "physical_pairing": "Exact physical model/full-state fingerprints must match existing source canonical states captured at256 rendering. Camera resolution differs explicitly; cross-model pixel equality is not claimed.",
        "reset_contract": "Same seed and settling actions as the frozen shared source protocol, including gripper-1. This differs from the original DMPEL evaluation's five all-zero settling actions.",
        "controller": "Relative OSC; use_delta=True only after exact common-reset verification",
        "sampling_contract": "Original CUDA-half GMM sampling versus explicit float32 canonical mixture sampling/log-density. Equal integer seeds are not equal latent samples across these numerical sampling laws.",
        "weight_training_scope": "Author pretraining cfg declares LIBERO90; no independent reconstruction of training membership. Full lifelong DMPEL and published scores are not reproduced.",
        "coverage_review": "Report all48 complete cases, per-task outcomes and source headroom before choosing any future proposer. No training or main-target launch is authorized by this script.",
        "interrupted_attempts": "A started case without a final result requires explicit infrastructure review; never silently resample a failed/incomplete physical attempt",
        "main_target_outcomes_seen": 0, "independent_confirmation_outcomes_seen": 0,
        "runtime_port": "Torch2.9.1, Transformers4.51.3 eager CLIP, timm1.0.15, original robomimic0.2.0; deterministic position-ID persistence compatibility handled explicitly. Linux EGL-probe is not available in this Windows policy worker; rendering runs in the separately validated simulator environment."}
    write(path, plan)
    print("REGISTERED_DMPEL_SOURCE48_INITIALIZATION_PROBE", flush=True)
    return plan


def gpu_schema_check(client, plan):
    image = (np.arange(128 * 128 * 3).reshape(128, 128, 3) % 251).astype(np.uint8)
    payload = {"agentview_image": image, "robot0_eye_in_hand_image": image[::-1].copy(),
        "robot0_joint_pos": np.asarray([0, -.5, 0, -2, 0, 1.5, .8]), "robot0_gripper_qpos": np.asarray([.02, -.02])}
    path = WORK / "gpu_schema_observation.npz"
    np.savez_compressed(path, **payload)
    language = "put the butter at the back in the top drawer of the cabinet and close it"
    parameters, repeats = {}, {}
    for variant in plan["variants"]:
        actions = []
        for _ in range(2):
            client.request("select", variant=variant, seed=97500)
            result = client.request("query", observation_file=str(path), language=language, include_parameters=True)
            actions.append(np.asarray(result["action"]))
            parameters[variant] = result["distribution_parameters"]
        difference = float(np.max(np.abs(actions[0] - actions[1])))
        assert difference == 0
        repeats[variant] = difference
    comparisons = {key: float(np.max(np.abs(np.asarray(parameters["native_half"][key]) -
                                            np.asarray(parameters["canonical_float32"][key]))))
                   for key in parameters["native_half"]}
    assert max(comparisons.values()) <= 1e-6, comparisons
    write(WORK / "gpu_schema_check.json", {"checked_utc": datetime.now(UTC).isoformat(),
        "status": "passed_actual_gpu_synthetic_reset_and_sampler_parameter_checks", "repeat_action_differences": repeats,
        "sampler_parameter_differences": comparisons, "synthetic_observations": True, "physical_episodes_added": 0,
        "metadata": client.metadata, "registration_sha256": sha(WORK / "registration.json")})


def reset_shared(setting, index, primary_plan):
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from xvla_benchmark_v4 import controller_observation

    assert sha(ROOT / setting["bddl"]) == setting["bddl_sha256"]
    assert sha(ROOT / setting["init_file"]) == setting["init_sha256"]
    states = torch.load(ROOT / setting["init_file"], weights_only=False)
    seed = primary_plan["policy_seeds"][0]
    np.random.seed(seed)
    random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=str(ROOT / setting["bddl"]), camera_heights=128, camera_widths=128,
                            camera_names=["agentview", "robot0_eye_in_hand"], hard_reset=False)
    env.seed(seed)
    env.reset()
    raw = env.set_init_state(states[index])
    for _ in range(primary_plan["settling_steps"]):
        raw, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    for robot in env.env.robots:
        robot.controller.use_delta = False
    return env, controller_observation(env, raw)


def episode(setting, index, variant, plan, primary_plan, client):
    import imageio.v2 as imageio
    from paired_observation_v4 import canonicalize, physical_fingerprints
    from xvla_benchmark_v4 import diagnostic_subgoals

    key = f"source90_task{setting['task_id']}_init{index}_noise{plan['policy_seed']}_{variant}"
    path = WORK / "results" / (key + ".json")
    if path.exists():
        row = read(path)
        assert row["registration_sha256"] == sha(WORK / "registration.json")
        assert row["trace_sha256"] == sha(path.with_suffix(".npz"))
        assert row["video_sha256"] == sha(WORK / row["video"])
        return row
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt = WORK / "attempts" / (key + ".json")
    assert not attempt.exists(), "Unfinished physical attempt requires review before any retry"
    write(attempt, {"started_utc": datetime.now(UTC).isoformat(), "key": key,
                    "status": "starting_source_case_not_a_completed_result", "registration_sha256": sha(WORK / "registration.json")})
    started = time.perf_counter()
    env, raw = reset_shared(setting, index, primary_plan)
    try:
        old = read(PRIMARY / "canonical" / f"libero_90_task{setting['task_id']}_original_init{index}.json")
        model_sha, state_sha = physical_fingerprints(env)
        assert model_sha == old["model_sha256"] and state_sha == old["full_state_sha256"], "Source reset differs physically from earlier benchmark"
        raw, pairing = canonicalize(env, raw, setting, index, WORK / "canonical")
        joint_path = WORK / "canonical" / f"task{setting['task_id']}_init{index}_joint.npy"
        if joint_path.exists():
            np.testing.assert_array_equal(np.load(joint_path), raw["robot0_joint_pos"])
        else:
            np.save(joint_path, raw["robot0_joint_pos"])
        for robot in env.env.robots:
            robot.controller.use_delta = True
        client.request("select", variant=variant, seed=plan["policy_seed"])
        initial = np.asarray(env.get_sim_state()).copy()
        actions, states, rewards, subgoals, features, log_probabilities, timings = [], [], [], [], [], [], []
        queries = {name: [] for name in SENSORS}
        video = WORK / "videos" / (key + ".mp4")
        video.parent.mkdir(parents=True, exist_ok=True)
        temporary_video = video.with_name(video.stem + ".working.mp4")
        assert not temporary_video.exists(), "Do not overwrite an interrupted video"
        with imageio.get_writer(temporary_video, fps=20, quality=7, macro_block_size=1) as writer:
            for _ in range(plan["horizon"]):
                observed = {name: np.asarray(raw[name]).copy() for name in SENSORS}
                temporary = WORK / "current_observation.npz"
                np.savez_compressed(temporary, **observed)
                result = client.request("query", observation_file=str(temporary), language=setting["language"])
                action = np.asarray(result["action"], dtype=np.float32)
                assert action.shape == (7,) and np.isfinite(action).all()
                features.append(np.load(result["spatial_feature_file"]).copy())
                for name in SENSORS:
                    queries[name].append(observed[name])
                if variant == "canonical_float32":
                    log_probabilities.append(result["log_probability"])
                raw, reward, done, _ = env.step(action)
                actions.append(action)
                states.append(np.concatenate([raw["robot0_eef_pos"], raw["robot0_eef_quat"], raw["robot0_gripper_qpos"]]))
                rewards.append(reward)
                subgoals.append(diagnostic_subgoals(env))
                timings.append({key: result[key] for key in ["query_seconds", "peak_cuda_allocated_bytes"]})
                writer.append_data(np.concatenate([raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"][::-1, ::-1]], axis=1))
                if env.check_success() or done:
                    break
        success = bool(env.check_success())
        arrays = {"actions": np.asarray(actions), "robot_states": np.asarray(states), "rewards": np.asarray(rewards),
                  "diagnostic_subgoals": np.asarray(subgoals), "initial_sim_state": initial,
                  "frozen_spatial_features": np.asarray(features),
                  **{"query_" + name: np.stack(values) for name, values in queries.items()}}
        if variant == "canonical_float32":
            arrays["behavior_log_probabilities"] = np.asarray(log_probabilities, dtype=np.float32)
            assert np.isfinite(arrays["behavior_log_probabilities"]).all()
        temporary_trace = path.with_suffix(".working.npz")
        with temporary_trace.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary_trace.replace(path.with_suffix(".npz"))
        temporary_video.replace(video)
        row = {"key": key, "task_id": setting["task_id"], "language": setting["language"], "state_index": index,
               "policy_seed": plan["policy_seed"], "variant": variant, "success": success, "complete_task": True,
               "steps": len(actions), "success_by400": success and len(actions) <= 400,
               "success_by520": success and len(actions) <= 520,
               "registration_sha256": sha(WORK / "registration.json"), "trace_sha256": sha(path.with_suffix(".npz")),
               "video": video.relative_to(WORK).as_posix(), "video_sha256": sha(video),
               "paired_observation": pairing, "joint_snapshot_sha256": sha(joint_path),
               "native_model_metadata": client.metadata, "query_timings": timings,
               "wall_seconds": time.perf_counter() - started, "completed_utc": datetime.now(UTC).isoformat(),
               "main_target_outcomes_seen": 0, "learned_motor_gain_claim": False}
        write(path, row)
        print("DMPEL_SOURCE_INITIALIZATION_COMPLETE", key, success, len(actions), flush=True)
        return row
    finally:
        env.close()
        gc.collect()


def summarize(plan):
    rows = [read(WORK / "results" / f"source90_task{task}_init{index}_noise{plan['policy_seed']}_{variant}.json")
            for task in plan["source_task_ids"] for index in plan["states"] for variant in plan["variants"]]
    assert len(rows) == 48
    for row in rows:
        assert row["registration_sha256"] == sha(WORK / "registration.json")
    by_variant = {}
    for variant in plan["variants"]:
        subset = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {"episodes": 24, "successes_by600": sum(row["success"] for row in subset),
            "successes_by520": sum(row["success_by520"] for row in subset),
            "successes_by400": sum(row["success_by400"] for row in subset),
            "successful_tasks_by520": len({row["task_id"] for row in subset if row["success_by520"]}),
            "native_queries": sum(row["steps"] for row in subset),
            "inference_seconds": sum(timing["query_seconds"] for row in subset for timing in row["query_timings"])}
    comparisons = {}
    bootstrap = np.random.default_rng(20261020).integers(12, size=(20000, 12))
    for field in ["success_by400", "success_by520", "success"]:
        differences = []
        wins = losses = 0
        for task in plan["source_task_ids"]:
            task_differences = []
            for index in plan["states"]:
                pair = {row["variant"]: row for row in rows if row["task_id"] == task and row["state_index"] == index}
                for key in ["model_sha256", "full_state_sha256", "canonical_observation_sha256"]:
                    assert pair["native_half"]["paired_observation"][key] == pair["canonical_float32"]["paired_observation"][key]
                assert pair["native_half"]["joint_snapshot_sha256"] == pair["canonical_float32"]["joint_snapshot_sha256"]
                difference = int(pair["canonical_float32"][field]) - int(pair["native_half"][field])
                wins += difference > 0
                losses += difference < 0
                task_differences.append(difference)
            differences.append(np.mean(task_differences))
        values = np.asarray(differences)
        comparisons[field] = {"canonical_minus_native_task_macro": float(values.mean()),
            "paired_context_improvements": wins, "paired_context_harms": losses,
            "exploratory_task_cluster_bootstrap95": np.quantile(values[bootstrap].mean(axis=1), [.025, .975]).tolist()}
    write(WORK / "summary.json", {"status": "complete_source_only_initialization_precision_probe_requires_review",
        "episodes": 48, "physical_initial_states": 24, "semantic_tasks": 12, "by_variant": by_variant,
        "precision_configuration_contrasts": comparisons,
        "inference_limits": "Source-descriptive,12 task clusters and one stochastic rollout per state/mode. Equal integer seeds are not shared latent samples across precision ports. Degenerate observed bootstrap intervals do not establish zero population uncertainty. These differences are not RL gains.",
        "cases": rows, "registration_sha256": sha(WORK / "registration.json"),
        "main_target_outcomes_seen": 0, "learned_motor_gain_claim": False, "RSS_readiness_claim": False})


def collect():
    from local_run_lock_v4 import exclusive
    from probe_source_reference_v4 import settings
    from dmpel_probe_client_v6 import ModelClient
    plan = register()
    primary = read(PRIMARY / "protocol.json")
    assert sha(PRIMARY / "protocol.json") == plan["source_protocol_sha256"]
    chosen = settings(primary)
    assert [row["task_id"] for row in chosen] == plan["source_task_ids"]
    with exclusive("dmpel_source_probe_v6"), exclusive("robotics_gpu"):
        client = ModelClient()
        try:
            gpu_schema_check(client, plan)
            for setting in chosen:
                for index in plan["states"]:
                    order = sorted(plan["variants"], key=lambda variant: hashlib.sha256(f"{setting['task_id']}:{index}:{variant}".encode()).hexdigest())
                    for variant in order:
                        episode(setting, index, variant, plan, primary, client)
        finally:
            client.close()
    summarize(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["register", "collect"])
    stage = parser.parse_args().stage
    register() if stage == "register" else collect()

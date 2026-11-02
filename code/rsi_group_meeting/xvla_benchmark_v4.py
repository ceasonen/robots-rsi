"""Official LIBERO/Pro complete episodes with the author's actual LoRA bank.

Saved initializations, noise seeds and outcomes are paired across all candidates.
Formal source, calibration and test observations have disjoint state indices.
"""

import argparse
import gc
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from benchmark_v4 import ROOT, digest, write_json
from paired_observation_v4 import canonicalize
from xvla_plan_v4 import OUT, make_plan


def context_key(setting, index, seed):
    return f"{setting['suite']}_task{setting['task_id']}_{setting['perturbation']}_init{index}_seed{seed}"


def controller_observation(env, raw):
    controller = env.env.robots[0].controller
    result = dict(raw)
    result["robo_pos"] = np.asarray(controller.ee_pos).copy()
    rotation = np.asarray(controller.ee_ori_mat)
    result["robo_ori"] = np.concatenate([rotation[:, 0], rotation[:, 1]]).copy()
    return result


def diagnostic_subgoals(env):
    """Read-only official predicate diagnostics, NEVER actor/router inputs."""
    task = env.env
    definitions = task.parsed_problem["goal_state"]
    values = np.asarray([bool(task._eval_predicate(goal)) for goal in definitions], dtype=bool)
    assert len(values) and bool(values.all()) == bool(env.check_success())
    return values


def freeze_sources():
    """A resume must use the same physical/predictive implementation."""
    files = [
        "benchmark_v4.py",
        "xvla_plan_v4.py",
        "xvla_policy_v4.py",
        "xvla_benchmark_v4.py",
        "xvla_selection_v4.py",
        "research_controls_v4.py",
        "paired_observation_v4.py",
        "local_run_lock_v4.py",
        "local_repair_gate_v4.py",
        "local_repair_paths_v4.py",
        "v4/xvla/execution_backend.json",
    ]
    folder = OUT / "source_snapshots"
    manifest = {name: digest(ROOT / "rsi_group_meeting" / name) for name in files}
    path = folder / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == manifest, (
            "Formal implementation changed: use a new experiment or an explicit audited amendment"
        )
    else:
        assert not list((OUT / "results").rglob("*.json")), "Cannot snapshot after outcomes"
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            saved = folder / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes((ROOT / "rsi_group_meeting" / name).read_bytes())
        write_json(path, manifest)
        # Preserve registration history without forcing a rerun of unchanged
        # physical episodes when a separately reported SOURCE-ONLY motor
        # proposer is revised. The final extension's actors/plan are hash-bound
        # by its independent gate before any primary heldout action.
        historical = folder / "registration_metadata"
        historical.mkdir(exist_ok=True)
        metadata = {}
        for name in ["provenance_scope.json", "local_repair_extension_plan.json"]:
            target = historical / name
            target.write_bytes((OUT / name).read_bytes())
            metadata[name] = digest(target)
        write_json(historical / "manifest.json", metadata)


def reset_env(setting, index, plan):
    from libero.libero.envs import OffScreenRenderEnv

    assert digest(ROOT / setting["bddl"]) == setting["bddl_sha256"]
    assert digest(ROOT / setting["init_file"]) == setting["init_sha256"]
    states = torch.load(ROOT / setting["init_file"], weights_only=False)
    assert 0 <= index < len(states)
    np.random.seed(plan["policy_seeds"][0])
    random.seed(plan["policy_seeds"][0])
    env = OffScreenRenderEnv(
        bddl_file_name=str(ROOT / setting["bddl"]),
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        hard_reset=False,
    )
    env.seed(plan["policy_seeds"][0])
    env.reset()
    raw = env.set_init_state(states[index])
    for _ in range(plan["settling_steps"]):
        raw, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    for robot in env.env.robots:
        robot.controller.use_delta = False
    return env, controller_observation(env, raw)


def run_episode(
    setting,
    index,
    seed,
    variant,
    plan,
    policy,
    phase,
    smoke_steps=None,
    decision_hook=None,
    shared_sensor_root=None,
):
    integration = phase == "integration"
    integration_root = OUT / "integration" / f"protocol{plan['protocol_version']}"
    folder = (
        (integration_root / ("smoke31" if smoke_steps else "complete"))
        if integration
        else OUT / "results" / phase
    )
    folder.mkdir(parents=True, exist_ok=True)
    stem = context_key(setting, index, seed)
    key = stem + "_" + variant
    receipt_path = folder / (key + ".json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        assert receipt["protocol_sha256"] == digest(OUT / "protocol.json")
        assert receipt["trace_sha256"] == digest(folder / (key + ".npz"))
        print("REUSE", key, flush=True)
        return receipt
    policy.reset(variant)
    policy.record_training_queries = phase == "source"
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    env, raw = reset_env(setting, index, plan)
    sensor_root = shared_sensor_root or (integration_root if integration else OUT)
    raw, paired_observation = canonicalize(env, raw, setting, index, sensor_root / "canonical")
    initial_sha = hashlib.sha256(np.asarray(env.get_sim_state()).tobytes()).hexdigest()
    observation_sha = hashlib.sha256(
        raw["agentview_image"].tobytes()
        + raw["robot0_eye_in_hand_image"].tobytes()
        + np.asarray(raw["robo_pos"]).tobytes()
        + np.asarray(raw["robo_ori"]).tobytes()
        + setting["language"].encode()
    ).hexdigest()
    context_dir = sensor_root / "contexts"
    context_dir.mkdir(parents=True, exist_ok=True)
    sensor_stem = f"{setting['suite']}_task{setting['task_id']}_{setting['perturbation']}_init{index}"
    context_path = context_dir / (sensor_stem + ".npz")
    context_receipt = context_path.with_suffix(".json")
    if context_receipt.exists():
        previous = json.loads(context_receipt.read_text(encoding="utf8"))
        assert previous["initial_sim_sha256"] == initial_sha
        assert previous["observation_sha256"] == observation_sha
        assert previous["trace_sha256"] == digest(context_path)
        features = np.load(context_path)["features"]
    else:
        assert not policy.full_reference, "Collect the bank before the separate full reference"
        features = policy.context(raw, setting["language"])
        assert np.isfinite(features).all()
        np.savez_compressed(
            context_path,
            features=features,
            agentview=raw["agentview_image"],
            wrist=raw["robot0_eye_in_hand_image"],
        )
        write_json(
            context_receipt,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "initial_sim_sha256": initial_sha,
                "observation_sha256": observation_sha,
                "trace_sha256": digest(context_path),
                "feature_encoder": "frozen foundation with all adapters disabled; first observation only",
            },
        )
    if phase == "heldout":
        if decision_hook is None:
            from local_repair_gate_v4 import freeze_optional_choices
            from xvla_selection_v4 import freeze_choices

            freeze_choices(setting, index, seed, features, initial_sha, observation_sha)
            freeze_optional_choices(setting, index, seed, features, initial_sha, observation_sha)
        else:
            decision_hook(setting, index, seed, features, initial_sha, observation_sha)
    video_phase = (
        (f"protocol{plan['protocol_version']}/" + ("smoke31" if smoke_steps else "complete"))
        if integration
        else phase
    )
    video = OUT / ("integration/videos" if integration else "videos") / video_phase / (key + ".mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        video, fps=20, codec="libx264", ffmpeg_params=["-crf", "21", "-movflags", "+faststart"]
    )
    actions, robot_states, latencies, rewards = [], [], [], []
    subgoal_definitions = env.env.parsed_problem["goal_state"]
    subgoals = [diagnostic_subgoals(env)]
    query_steps, query_images, query_proprio, query_predictions = [], [], [], []
    first_control_utc = None
    success = False
    limit = smoke_steps or setting["max_steps"]
    try:
        for step in range(limit):
            writer.append_data(
                np.concatenate([raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"]], axis=1)
            )
            tick = time.perf_counter()
            record_query = phase == "source" and not policy.queue
            if record_query:
                query_steps.append(step)
                query_images.append(
                    np.stack([raw["agentview_image"], raw["robot0_eye_in_hand_image"]]).copy()
                )
            action = policy.step(raw, setting["language"])
            if record_query:
                query_proprio.append(policy.last_query_proprio.copy())
                query_predictions.append(policy.last_query_prediction.copy())
            latencies.append(time.perf_counter() - tick)
            if step == 0:
                first_control_utc = datetime.now(UTC).isoformat()
            raw, reward, _done, _info = env.step(action)
            raw = controller_observation(env, raw)
            actions.append(action)
            rewards.append(float(reward))
            robot_states.append(
                np.concatenate([raw["robot0_eef_pos"], raw["robot0_eef_quat"], raw["robot0_gripper_qpos"]])
            )
            success = bool(env.check_success())
            subgoals.append(diagnostic_subgoals(env))
            if step % 60 == 0:
                print("X_PROGRESS", key, step, round(time.perf_counter() - started, 1), flush=True)
            if success:
                break
        writer.append_data(
            np.concatenate([raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"]], axis=1)
        )
        writer.close()
        writer = None
        trace = folder / (key + ".npz")
        np.savez_compressed(
            trace,
            actions=actions,
            robot_states=robot_states,
            decision_times=latencies,
            rewards=rewards,
            initial_features=features,
            diagnostic_subgoals=np.asarray(subgoals, dtype=bool),
            query_steps=np.asarray(query_steps, dtype=np.int32),
            query_images=np.asarray(query_images, dtype=np.uint8),
            query_proprio=np.asarray(query_proprio, dtype=np.float32),
            query_predictions=np.asarray(query_predictions, dtype=np.float32),
        )
        primary_wall = time.perf_counter() - started
        primary_inferences = policy.inferences
        native_horizon = 900 if setting["suite"] == "libero_10" else 800
        continuation = {
            "performed": False,
            "success": success,
            "additional_steps": 0,
            "native_cap": native_horizon,
            "independent_sample": False,
        }
        if phase in ["heldout", "reference"] and not success:
            extra_started = time.perf_counter()
            extra_video = video.with_name(video.stem + "_native_continuation.mp4")
            writer = imageio.get_writer(
                extra_video, fps=20, codec="libx264", ffmpeg_params=["-crf", "21", "-movflags", "+faststart"]
            )
            extra_actions, extra_states, extra_success = [], [], False
            extra_subgoals = [diagnostic_subgoals(env)]
            for step in range(limit, native_horizon):
                writer.append_data(
                    np.concatenate(
                        [raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"]], axis=1
                    )
                )
                action = policy.step(raw, setting["language"])
                raw, _reward, _done, _info = env.step(action)
                raw = controller_observation(env, raw)
                extra_actions.append(action)
                extra_states.append(
                    np.concatenate(
                        [raw["robot0_eef_pos"], raw["robot0_eef_quat"], raw["robot0_gripper_qpos"]]
                    )
                )
                extra_success = bool(env.check_success())
                extra_subgoals.append(diagnostic_subgoals(env))
                if step % 120 == 0:
                    print("X_NATIVE_DIAGNOSTIC", key, step, flush=True)
                if extra_success:
                    break
            writer.append_data(
                np.concatenate([raw["agentview_image"][::-1, ::-1], raw["robot0_eye_in_hand_image"]], axis=1)
            )
            writer.close()
            writer = None
            extra_trace = trace.with_name(trace.stem + "_native_continuation.npz")
            np.savez_compressed(
                extra_trace,
                actions=extra_actions,
                robot_states=extra_states,
                diagnostic_subgoals=np.asarray(extra_subgoals, dtype=bool),
            )
            continuation.update(
                {
                    "performed": True,
                    "success": extra_success,
                    "additional_steps": len(extra_actions),
                    "wall_seconds": time.perf_counter() - extra_started,
                    "trace": extra_trace.relative_to(OUT).as_posix(),
                    "trace_sha256": digest(extra_trace),
                    "video": extra_video.relative_to(OUT).as_posix(),
                    "video_sha256": digest(extra_video),
                }
            )
        receipt = {
            "key": key,
            "context_key": stem,
            "phase": phase,
            "suite": setting["suite"],
            "task_id": setting["task_id"],
            "task_name": setting["task_name"],
            "perturbation": setting["perturbation"],
            "state_index": index,
            "policy_seed": seed,
            "variant": variant,
            "motor_training_seed": getattr(policy, "motor_training_seed", None),
            "local_actor_training_receipt_sha256": getattr(policy, "training_receipt_sha256", None),
            "success": success,
            "steps": len(actions),
            "horizon": limit,
            "complete_task": smoke_steps is None,
            "source_query_RGB_for_optional_local_repair": phase == "source",
            "diagnostic_subgoal_definitions": subgoal_definitions,
            "privileged_subgoals_for_posthoc_diagnostics_only": True,
            "integration_only": integration,
            "inferences": policy.inferences,
            "primary_inferences": primary_inferences,
            "wall_seconds": time.perf_counter() - started,
            "primary_wall_seconds": primary_wall,
            "native_horizon_diagnostic": continuation,
            "simulated_seconds": (len(actions) + plan["settling_steps"]) / plan["control_hz"],
            "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "initial_sim_sha256": initial_sha,
            "paired_observation": paired_observation,
            "observation_sha256": observation_sha,
            "initial_features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "trace_sha256": digest(trace),
            "video_sha256": digest(video),
            "video": video.relative_to(OUT).as_posix(),
            "protocol_sha256": digest(OUT / "protocol.json"),
            "learning_protocol_sha256": digest(OUT / "learning_protocol.json"),
            "runner_sha256": digest(Path(__file__)),
            "policy_code_sha256": digest(ROOT / "rsi_group_meeting/xvla_policy_v4.py"),
            "asset_receipt_sha256": digest(OUT / "asset_receipt.json"),
            "actual_worker_asset_verification": policy.asset_verification,
            "first_control_utc": first_control_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        write_json(receipt_path, receipt)
        print(
            "X_COMPLETED",
            key,
            "success",
            success,
            "steps",
            len(actions),
            "wall",
            round(receipt["wall_seconds"], 1),
            flush=True,
        )
        return receipt
    finally:
        if writer is not None:
            writer.close()
        env.close()
        del env, raw
        # Fresh render/simulator objects contain reference cycles on Windows.
        # Release dead contexts and unused CUDA cache between episodes while
        # keeping the native model and inference arithmetic unchanged.
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["smoke", "integration", "source", "validation", "heldout", "reference"]
    )
    parser.add_argument(
        "--limit-contexts", type=int, help="Resumable scheduling limit, never change population"
    )
    parser.add_argument(
        "--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
    )
    parser.add_argument("--full-reference-integration", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    from xvla_selection_v4 import freeze_protocol

    freeze_protocol()
    assert (OUT / "asset_receipt.json").exists(), "Public checkpoint verification must finish first"
    from xvla_policy_v4 import XPolicy

    phase = "integration" if args.stage in ["smoke", "integration"] else args.stage
    if phase != "integration":
        freeze_sources()
    assert not args.full_reference_integration or phase == "integration"
    full = args.stage == "reference" or args.full_reference_integration
    policy = XPolicy(full_reference=full)
    if phase == "integration":
        settings = [s for s in plan["settings"] if s["perturbation"] == "original" and s["task_id"] == 0]
        indices, seeds = [plan["integration_state_index"]], plan["policy_seeds"][:1]
    elif phase in ["source", "validation"]:
        settings = [
            s
            for s in plan["settings"]
            if s["suite"] in plan["source_suites"] and s["perturbation"] in plan["source_perturbations"]
        ]
        indices = (
            plan["source_state_indices"] if phase == "source" else plan["source_validation_state_indices"]
        )
        seeds = plan["policy_seeds"]
    else:
        from local_repair_gate_v4 import verify_decision

        verify_decision()
        assert (OUT / "models/freeze_receipt.json").exists(), (
            "Freeze all source-only routers before test outcomes"
        )
        settings = [s for s in plan["settings"] if s["suite"] in plan["heldout_suites"]]
        indices, seeds = plan["heldout_state_indices"], plan["policy_seeds"]
    if args.suite:
        settings = [s for s in settings if s["suite"] == args.suite]
    if phase in ["source", "validation"]:
        # Outcome-independent diverse early scheduling. Bank loading and
        # learning retain the original canonical task order.
        settings = sorted(
            settings,
            key=lambda item: hashlib.sha256(
                ("rsi-source-order-76110|" + item["task_name"]).encode()
            ).hexdigest(),
        )
    variants = ["full_libero"] if full else ["base", "object", "goal"]
    contexts = [(s, i, seed) for s in settings for i in indices for seed in seeds]
    scheduled = 0
    for setting, index, seed in contexts:
        paths = [
            OUT / "results" / phase / (context_key(setting, index, seed) + "_" + v + ".json")
            for v in variants
        ]
        complete = all(p.exists() for p in paths)
        if not complete and args.limit_contexts is not None and scheduled >= args.limit_contexts:
            break
        for variant in variants:
            run_episode(
                setting,
                index,
                seed,
                variant,
                plan,
                policy,
                phase,
                smoke_steps=31 if args.stage == "smoke" else None,
            )
        if not complete:
            scheduled += 1
    print("X_STAGE_COMPLETE", args.stage, "scheduled", scheduled, flush=True)


if __name__ == "__main__":
    from local_run_lock_v4 import exclusive

    with exclusive("robotics_gpu"):
        main()

"""Canonical first sensor realization after a strictly verified common reset.

OpenGL can round a handful of 8-bit channels differently across render contexts.
We retain the original allowed first observation and give exactly those bytes
to every candidate/policy-noise replica. Physical/model hashes remain exact.
No privileged model array is passed to the policy or to the learned router.
"""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
from benchmark_v4 import digest, write_json

OBSERVATIONS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]
MODEL_FIELDS = [
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rgba",
    "geom_friction",
    "geom_contype",
    "geom_conaffinity",
    "dof_damping",
    "dof_frictionloss",
    "site_pos",
    "site_quat",
    "site_size",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "light_pos",
    "light_dir",
    "light_diffuse",
    "light_ambient",
    "light_specular",
    "mat_rgba",
    "mat_texid",
    "tex_rgb",
]
DATA_FIELDS = [
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
]
CONTRACT = {
    "version": 1,
    "raster_maximum_channel_difference": 1,
    "raster_maximum_changed_channels_both_cameras": 128,
    "proprioception": "byte-identical",
    "model_and_simulator": "exact SHA256",
    "canonical_first_frame": "Original real rendered observation saved before the first controlled action, shared across candidates and policy-noise replicas of that same official state. Subsequent observations are native closed-loop sensor readings.",
    "not_a_relaxed_policy_input_match": "Policy inputs at step zero become byte-identical; tolerance applies only to verifying that a new OpenGL render differs by negligible raster rounding at an exactly identical physical/model state.",
    "privileged_arrays": "Evaluation/reset audit only, never predictor or actor inputs",
}


def array_hash(values):
    result = hashlib.sha256()
    for name, value in values.items():
        value = np.asarray(value)
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def physical_fingerprints(env):
    model, data = env.env.sim.model, env.env.sim.data
    model_values = {key: np.asarray(getattr(model, key)) for key in MODEL_FIELDS}
    data_values = {key: np.asarray(getattr(data, key)) for key in DATA_FIELDS}
    data_values["flattened_state"] = np.asarray(env.get_sim_state())
    return array_hash(model_values), array_hash(data_values)


def canonicalize(env, raw, setting, index, directory):
    directory.mkdir(parents=True, exist_ok=True)
    key = f"{setting['suite']}_task{setting['task_id']}_{setting['perturbation']}_init{index}"
    path = directory / (key + ".npz")
    receipt_path = path.with_suffix(".json")
    model_sha, full_state_sha = physical_fingerprints(env)
    observations = OBSERVATIONS + [name for name in ["robo_pos", "robo_ori"] if name in raw]
    measured = {name: np.asarray(raw[name]) for name in observations}
    measured_sha = array_hash(measured)
    differences = {"maximum_channel_difference": 0, "changed_channels": 0}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        assert receipt["model_sha256"] == model_sha, "Reset changes physical/visual model arrays"
        assert receipt["full_state_sha256"] == full_state_sha, "Reset changes dynamic/control/warmstart state"
        assert receipt["snapshot_sha256"] == digest(path)
        with np.load(path) as stored:
            for name in observations:
                reference = stored[name]
                if name.endswith("image"):
                    delta = np.abs(measured[name].astype(np.int16) - reference.astype(np.int16))
                    differences["maximum_channel_difference"] = max(
                        differences["maximum_channel_difference"], int(delta.max())
                    )
                    differences["changed_channels"] += int(np.count_nonzero(delta))
                else:
                    assert array_hash({name: measured[name]}) == array_hash({name: reference}), (
                        f"Reset changes initial {name}"
                    )
        assert differences["maximum_channel_difference"] <= CONTRACT["raster_maximum_channel_difference"], (
            differences
        )
        assert differences["changed_channels"] <= CONTRACT["raster_maximum_changed_channels_both_cameras"], (
            differences
        )
    else:
        np.savez_compressed(path, **measured)
        receipt = {
            "key": key,
            "created_utc": datetime.now(UTC).isoformat(),
            "model_sha256": model_sha,
            "full_state_sha256": full_state_sha,
            "canonical_observation_sha256": measured_sha,
            "snapshot_sha256": digest(path),
            "contract": CONTRACT,
            "outcomes_seen": 0,
        }
        write_json(receipt_path, receipt)
    result = dict(raw)
    with np.load(path) as stored:
        for name in observations:
            result[name] = stored[name].copy()
    assert (
        array_hash({name: result[name] for name in observations}) == receipt["canonical_observation_sha256"]
    )
    return result, {
        "model_sha256": model_sha,
        "full_state_sha256": full_state_sha,
        "measured_observation_sha256": measured_sha,
        "canonical_observation_sha256": receipt["canonical_observation_sha256"],
        "render_rounding_difference": differences,
        "snapshot_created_utc": receipt["created_utc"],
    }

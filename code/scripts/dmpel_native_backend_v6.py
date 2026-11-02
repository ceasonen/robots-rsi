"""Author DMPEL source90 initialization, not a reproduced lifelong DMPEL run.

Construct native modules without redundant pretrained downloads, then load the
complete public checkpoint strictly. GPU queries call the original get_action.
CPU engineering explicitly disables its CUDA-only autocast context; this CPU
float32 port is not claimed identical to the author's CUDA float16 inference.
"""

import ast
import copy
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/DMPEL"
MODEL = ROOT / "data/models/dmpel_author_source90"
CLIP = ROOT / "data/models/dmpel_clip_config_tokenizer"
COMMIT = "ae05fcbeb5ab7fe4f1a16de72903e62689ba1eea"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_model_namespaces():
    # The author package registry eagerly imports unrelated lifelong agents.
    # Import only native dependency modules; their source files stay untouched.
    for name, relative in [("libero", "libero"), ("libero.lifelong", "libero/lifelong"),
                           ("libero.lifelong.models", "libero/lifelong/models"),
                           ("libero.lifelong.models.modules", "libero/lifelong/models/modules")]:
        assert name not in sys.modules, "Use a dedicated policy process, not the simulator's LIBERO namespace"
        module = types.ModuleType(name)
        module.__path__ = [str(REPO / relative)]
        sys.modules[name] = module


class NativeDMPELSource90:
    def __init__(self, device="cpu", threads=2):
        assert device in ["cpu", "cuda"]
        import numpy as np
        import torch
        import timm
        from accelerate import init_empty_weights
        from easydict import EasyDict
        from transformers import AutoModel, AutoTokenizer, CLIPConfig
        import robomimic.utils.obs_utils as ObsUtils
        import robomimic.utils.tensor_utils as TensorUtils

        torch.set_num_threads(threads)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        assert revision == COMMIT
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip()
        model_receipt = json.loads((MODEL / "asset_receipt.json").read_text())
        clip_receipt = json.loads((CLIP / "asset_receipt.json").read_text())
        for folder, receipt in [(MODEL, model_receipt), (CLIP, clip_receipt)]:
            assert all(sha(folder / row["file"]) == row["sha256"] for row in receipt["files"])
        config = json.loads((MODEL / "config.json").read_text())
        assert config["benchmark_name"] == "libero_90" and config["policy"]["policy_type"] == "BCFoundationTailPolicy"
        assert config["data"]["use_joint"] and not config["data"]["use_ee"]
        native_model_namespaces()
        native_module = importlib.import_module("libero.lifelong.models.bc_foundation_tail_policy")
        cfg = EasyDict(copy.deepcopy(config))
        cfg.device = device
        original_timm = timm.create_model
        clip_config = CLIPConfig.from_pretrained(str(CLIP), local_files_only=True)

        def construct_visual(name, *args, **kwargs):
            assert name == "vit_base_patch16_clip_224.openai" and not args
            assert kwargs.pop("pretrained") is True and not kwargs
            return original_timm(name, pretrained=False)

        def construct_text(name, *args, **kwargs):
            assert name == "openai/clip-vit-base-patch16" and not args and not kwargs
            # Original transformers4.21 CLIP used eager attention. Do not
            # silently select a newer SDPA backend when constructing its weights.
            return AutoModel.from_config(clip_config, attn_implementation="eager")

        with patch.object(timm, "create_model", side_effect=construct_visual), \
                patch.object(AutoModel, "from_pretrained", side_effect=construct_text), \
                init_empty_weights(include_buffers=False):
            self.policy = native_module.BCFoundationTailPolicy(cfg, cfg.shape_meta)
        checkpoint_path = MODEL / "multitask_model_ep10.pth"
        # EasyDict's public constructor is an inspected plain data container.
        with torch.serialization.safe_globals([EasyDict]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        assert checkpoint["cfg"] == config
        # Old CLIP serialized this deterministic buffer; newer Transformers
        # marks it nonpersistent. Retain and verify it rather than drop a key.
        embeddings = self.policy.language_encoder_spatial.model.embeddings
        source_positions = checkpoint["state_dict"]["language_encoder_spatial.model.embeddings.position_ids"]
        assert torch.equal(source_positions, torch.arange(77, dtype=torch.long).reshape(1, 77))
        assert torch.equal(embeddings.position_ids, source_positions)
        embeddings.register_buffer("position_ids", embeddings.position_ids, persistent=True)
        result = self.policy.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
        assert not result.missing_keys and not result.unexpected_keys
        self.policy = self.policy.to(device).eval().requires_grad_(False)
        assert all(parameter.device.type == device for parameter in self.policy.parameters())
        assert all(value.device.type == device for value in self.policy.buffers())
        self.tokenizer = AutoTokenizer.from_pretrained(str(CLIP), local_files_only=True)
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
        # Compile the author's exact preprocessing function without importing
        # simulator orchestration, its CUDA helpers or evaluation side effects.
        metric_path = REPO / "libero/lifelong/metric.py"
        tree = ast.parse(metric_path.read_text(encoding="utf8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "raw_obs_to_tensor_obs")
        code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(metric_path), "exec")
        namespace = {"torch": torch, "ObsUtils": ObsUtils, "TensorUtils": TensorUtils,
                     "safe_device": lambda tensor, device: tensor.to(device)}
        exec(code, namespace)
        self.preprocess = namespace["raw_obs_to_tensor_obs"]
        self.cfg, self.torch, self.np, self.device = cfg, torch, np, device
        self._tokens = {}
        modules = {str(Path(value.__file__).relative_to(REPO)).replace("\\", "/"): sha(Path(value.__file__))
                   for value in list(sys.modules.values()) if getattr(value, "__file__", None)
                   and Path(value.__file__).is_relative_to(REPO) and Path(value.__file__).is_file()}
        self.metadata = {"checkpoint_kind": "author_source90_pretraining_initialization_not_lifelong_expert_library",
            "checkpoint_sha256": sha(checkpoint_path), "model_revision": model_receipt["revision"],
            "native_repository_commit": revision, "native_module_sha256": modules,
            "author_metric_sha256": sha(metric_path), "backend_sha256": sha(Path(__file__)),
            "device": device, "parameters": sum(parameter.numel() for parameter in self.policy.parameters()),
            "native_inference_precision": "author_cuda_float16_autocast" if device == "cuda" else "explicit_cpu_float32_engineering_port",
            "constructor_contract": "Native classes; only initial pretrained downloads bypassed; complete author state loaded strictly with assign=True",
            "dependency_compatibility": "Original config/code specifies older Transformers and Torch; this is a documented dependency port, not a claim of bitwise reproduction of published scores",
            "clip_position_ids": "Verified exact arange77 constant; restored persistent status for complete strict state loading",
            "clip_text_attention": "Explicit eager backend matching original CLIP implementation family",
            "preprocessing_contract": "Original raw_obs_to_tensor_obs AST; explicit tensor.to(device) replaces availability-based fallback helper",
            "allowed_sensors": ["agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos"],
            "image_resolution": 128, "raw_simulator_images_rotated": False, "feedback_action_steps": 1,
            "temporal_history_steps": self.policy.max_seq_len, "action_distribution": "original sampled GMM; no extra clipping",
            "versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "transformers", "timm", "robomimic", "numpy", "accelerate"]},
            "checkpoint_training_data_membership_independently_verified": False,
            "physical_episodes_added": 0}

    def reset(self, seed):
        self.policy.reset()
        self.torch.manual_seed(seed)

    def query(self, raw, language):
        np, torch = self.np, self.torch
        assert all(np.asarray(raw[key]).shape == (128, 128, 3) and np.asarray(raw[key]).dtype == np.uint8
                   for key in ["agentview_image", "robot0_eye_in_hand_image"])
        assert np.asarray(raw["robot0_joint_pos"]).shape == (7,)
        assert np.asarray(raw["robot0_gripper_qpos"]).shape == (2,)
        if language not in self._tokens:
            tokens = self.tokenizer(text=[language], add_special_tokens=True, max_length=self.cfg.data.max_word_len,
                                    padding="max_length", return_attention_mask=True, return_tensors="pt")
            self._tokens[language] = {key: tokens[key][0] for key in ["attention_mask", "input_ids"]}
        data = self.preprocess([raw], self._tokens[language], self.cfg)
        if self.device == "cpu":
            def cpu_context(kind, **kwargs):
                assert kind == "cuda" and kwargs == {"dtype": torch.float16}
                return nullcontext()
            with patch.object(torch.amp, "autocast", side_effect=cpu_context):
                action = self.policy.get_action(data)
        else:
            action = self.policy.get_action(data)
        assert action.shape == (1, 7) and np.isfinite(action).all()
        return action[0].astype(np.float32)

"""Verify this lightweight review snapshot; no model, simulator or experiment import."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    manifest = read(ROOT / "REVIEW_MANIFEST.json")
    for item in manifest["files"]:
        path = ROOT / item["file"]
        path.resolve().relative_to(ROOT)
        assert path.is_file(), item["file"]
        assert path.stat().st_size == item["bytes"], item["file"]
        assert sha(path) == item["sha256"], item["file"]

    base = ROOT / "evidence/v40"
    registration = read(base / "registration.json")
    readout = read(base / "analysis/coverage_readout.json")
    assert sha(base / "registration.json") == readout["registration_sha256"]
    bound_sources = 0
    for relative, expected in registration["implementation_sha256"].items():
        path = ROOT / "code" / relative
        if path.is_file():
            assert sha(path) == expected, relative
            bound_sources += 1
    assert bound_sources == 53
    for path in (ROOT / "code").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path.relative_to(ROOT)))

    raw_files = list((base / "raw48_result_json").glob("*.json"))
    assert len(raw_files) == 48
    rows = {row["key"]: row for row in readout["rows"]}
    assert len(rows) == 48
    counts = Counter()
    conditions = defaultdict(dict)
    task_controls = settling = 0
    for path in raw_files:
        raw = read(path)
        key = raw["episode"]["key"]
        row = rows[key]
        assert sha(path) == row["result_sha256"], key
        assert raw["success_by520"] == row["success520"], key
        assert raw["success_by900"] == row["success900"], key
        assert raw["executed_physics_steps"] == row["task_controls"], key
        assert raw["policy_queries"] == row["policy_queries"], key
        assert raw["action_samples_completed"] == row["action_samples_completed"], key
        assert raw["physical_episodes"] == 1 and not raw["privileged_controller"], key
        assert not raw["policy_numerical_failure"] and not raw["physics_instability_terminated"], key
        counts[(row["partition"], row["actor"])] += int(row["success520"])
        conditions[row["context_id"]][row["actor"]] = row
        task_controls += raw["executed_physics_steps"]
        settling += raw["settling_controls"]
    assert len(conditions) == 16
    panels = readout["panels"]
    for group, panel in panels.items():
        for actor, values in panel["by520"]["actors"].items():
            assert counts[(group, actor)] == values["successes"]
    for actor, expected_harm in [("oldA_KL", 3), ("coverage_KL", 1)]:
        repair = harm = 0
        for group in conditions.values():
            assert set(group) == {"parent", "oldA_KL", "coverage_KL"}
            parent, child = group["parent"]["success520"], group[actor]["success520"]
            repair += int(not parent and child)
            harm += int(parent and not child)
        assert repair == 0 and harm == expected_harm
    cost = read(base / "root_completed_review/cost_increment.json")
    assert task_controls == cost["new_task_controls"] == 18069
    assert settling == cost["new_settling_controls"] == 480
    assert readout["work_demonstrated"] is False
    print(json.dumps({"verified_files": len(manifest["files"]),
                      "registered_original_sources": bound_sources,
                      "raw_result_files": 48, "conditions": 16,
                      "task_controls": task_controls, "settling_controls": settling,
                      "model_or_simulator_calls": 0,
                      "full_reproduction_claim": False}, ensure_ascii=False))


if __name__ == "__main__":
    verify()

"""Reproducible CLI for separate synthetic development and testing."""
import argparse
import json
from pathlib import Path
from .experiment import run_one
from .env import TASKS


def main():
    parser = argparse.ArgumentParser(description="Synthetic CPU teaching-demo smoke-test (NOT robot evidence)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="7,17,27")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--out", default="")
    a = parser.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    tasks = [s.strip() for s in a.tasks.split(",")]
    if len(set(seeds)) != len(seeds) or len(set(tasks)) != len(tasks):
        parser.error("Seeds and task names must be unique")
    if any(t not in TASKS for t in tasks):
        parser.error(f"Known tasks: {', '.join(TASKS)}")
    results = [run_one(a.episodes, seed, task) for task in tasks for seed in seeds]
    output = {"schema_version": 1, "results": results}
    blob = json.dumps(output, sort_keys=True, indent=2, allow_nan=False)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(blob + "\n", encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    main()

"""Matched teacher-episode experiments with explicit budget and leakage controls."""
from dataclasses import asdict
import numpy as np
from .env import TwoStageEnv, TASKS
from .policies import STYLES, controller, LinearStudent


def split_seeds(episodes, seed):
    """Disjoint train, validation, source-test, target-test episode seeds."""
    n_train = max(20, episodes // 2)
    # Draw unique deterministic seeds; no benchmark outcomes involved.
    p = np.random.default_rng(seed).permutation(np.arange(100000, 100000 + n_train + 3 * episodes))
    return {
        "train": p[:n_train],
        "validation": p[n_train:n_train + episodes],
        "source_test": p[n_train + episodes:n_train + 2 * episodes],
        "target_test": p[n_train + 2 * episodes:],
    }


def collect(env, style, seeds):
    teacher = controller(style, env.task)
    data = []
    successful = 0
    n_steps = 0
    for s in seeds:
        ok, episode, steps = env.run(teacher, int(s), "target")
        successful += int(ok)
        n_steps += steps
        data.extend(episode)
    return data, successful, n_steps


def fit(env, data):
    return LinearStudent(env.task).fit(data)


def evaluate(env, policy, seeds, split):
    return np.array([env.run(policy, int(s), split)[0] for s in seeds], dtype=np.int64)


def run_one(episodes=200, seed=7, task_name="forward"):
    if episodes < 20:
        raise ValueError("episodes must be at least 20")
    env = TwoStageEnv(TASKS[task_name])
    rng = np.random.default_rng(seed + 982451653)
    splits = split_seeds(episodes, seed)
    train = splits["train"]
    validation = splits["validation"]
    n_pilot = max(8, len(train) // 5)
    n_pilot = min(n_pilot + n_pilot % 2, len(train) - 2)
    shuffled = rng.permutation(train)
    pilot = {"fast": shuffled[:n_pilot // 2], "cautious": shuffled[n_pilot // 2:n_pilot]}
    remaining = shuffled[n_pilot:]
    pilot_data, pilot_successes, pilot_steps = {}, {}, {}
    validation_scores = {}
    parent = controller("parent", env.task)
    parent_val_source = evaluate(env, parent, validation, "source")
    for style in STYLES:
        data, successes, steps = collect(env, style, pilot[style])
        pilot_data[style] = data
        pilot_successes[style] = successes
        pilot_steps[style] = steps
        pilot_student = fit(env, data)
        val_target = evaluate(env, pilot_student, validation, "target")
        val_source = evaluate(env, pilot_student, validation, "source")
        # Predeclared score, not chosen from held-out tests.
        validation_scores[style] = float(val_target.mean() - (parent_val_source & (1 - val_source)).mean())
    selected = max(STYLES, key=lambda style: (validation_scores[style], -STYLES.index(style)))
    opposite = next(s for s in STYLES if s != selected)

    allocations = {
        "fixed_fast": ["fast"] * len(train),
        "fixed_cautious": ["cautious"] * len(train),
        "random_mixture": list(rng.choice(STYLES, size=len(train))),
    }
    def get_arm(style):
        # Feedback controls *new demonstration generation* after pilot feedback.
        # Both pilot styles contribute to final student training.
        data = list(pilot_data["fast"]) + list(pilot_data["cautious"])
        success = sum(pilot_successes.values())
        controls = sum(pilot_steps.values())
        more, ok, steps = collect(env, style, remaining)
        return data + more, int(success + ok), int(controls + steps)

    feedback_data, feedback_success, feedback_controls = get_arm(selected)
    shuffled_data, shuffled_success, shuffled_controls = get_arm(opposite)
    # Deliberately anti-correlated with real feedback; a falsification control,
    # not claimed as a statistically independent shuffled-feedback estimator.
    datasets = {
        "feedback_selected": (feedback_data, feedback_success, feedback_controls),
        "feedback_inverted_control": (shuffled_data, shuffled_success, shuffled_controls),
    }
    for name, styles in allocations.items():
        # One seed per teacher episode, one teacher trajectory per requested style.
        data, successes, controls = [], 0, 0
        for style, s in zip(styles, train):
            d, k, c = collect(env, str(style), [int(s)])
            data.extend(d)
            successes += k
            controls += c
        datasets[name] = data, successes, controls

    parent_t = evaluate(env, parent, splits["target_test"], "target")
    parent_s = evaluate(env, parent, splits["source_test"], "source")
    results = {}
    for name, (data, teacher_success, control_steps) in datasets.items():
        student = fit(env, data)
        target = evaluate(env, student, splits["target_test"], "target")
        source = evaluate(env, student, splits["source_test"], "source")
        results[name] = {
            "target_success": int(target.sum()),
            "source_success": int(source.sum()),
            "target_repair": int(((parent_t == 0) & (target == 1)).sum()),
            "target_harm": int(((parent_t == 1) & (target == 0)).sum()),
            "source_repair": int(((parent_s == 0) & (source == 1)).sum()),
            "source_harm": int(((parent_s == 1) & (source == 0)).sum()),
            "teacher_success": teacher_success,
            "teacher_episodes": int(len(train)),
            "teacher_control_steps": control_steps,
            "training_action_samples": len(data),
        }
    return {
        "environment": "1D synthetic toy abstraction; NOT MuJoCo/LIBERO-Pro",
        "task": task_name, "task_spec": asdict(env.task), "seed": seed,
        "episodes_per_eval_panel": episodes, "teacher_episodes_per_method": int(len(train)),
        "selection_validation_episodes_per_style_and_panel": episodes,
        "selection_extra_student_fits": 2,
        "selected_style": selected, "pilot_scores": validation_scores,
        "parent_target_success": int(parent_t.sum()),
        "parent_source_success": int(parent_s.sum()),
        "results": results,
        "limitations": [
            "No physical simulation, pretrained model, image inputs, or original teacher trajectory",
            "All methods receive equal teacher episode counts, NOT equal action targets or compute",
            "Feedback arms perform extra pilot training and validation; include this cost in comparisons",
            "Feedback inverted control is a stress test, not a permutation significance test",
            "Successful toy results cannot establish robotic improvement or publication significance",
        ],
    }

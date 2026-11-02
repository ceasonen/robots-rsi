"""Minimal deterministic two-stage, 1D manipulation *abstraction*.

This is not a physics engine and carries no LIBERO performance claims.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    name: str
    pickup: float
    drop: float
    horizon: int
    disturbance: float


TASKS = {
    "forward": TaskSpec("forward", pickup=0.35, drop=0.85, horizon=8, disturbance=0.58),
    "reverse": TaskSpec("reverse", pickup=0.56, drop=0.16, horizon=10, disturbance=0.48),
    "tight": TaskSpec("tight", pickup=0.29, drop=0.78, horizon=8, disturbance=0.40),
}


@dataclass
class State:
    x: float
    phase: int
    step: int
    disturbance: float


class TwoStageEnv:
    def __init__(self, task: TaskSpec):
        self.task = task

    def reset(self, seed: int, split: str) -> State:
        if split not in ("source", "target"):
            raise ValueError(split)
        rng = np.random.default_rng(seed)
        x = float(rng.uniform(0.0, 0.18))
        d = float(rng.uniform(-self.task.disturbance, self.task.disturbance)) if split == "target" else 0.0
        return State(x=x, phase=0, step=0, disturbance=d)

    def observe(self, state: State) -> np.ndarray:
        # x and phase are available to both controllers. Disturbance is hidden.
        return np.array([state.x, float(state.phase), state.step / self.task.horizon, 1.0], dtype=np.float64)

    def step(self, state: State, action: float):
        if not np.isfinite(action):
            raise ValueError("Nonfinite action")
        state.x = float(np.clip(state.x + np.clip(action, -0.12, 0.12), 0.0, 1.0))
        state.step += 1
        if state.phase == 0 and abs(state.x - self.task.pickup) < 0.055:
            state.phase = 1
            state.x = float(np.clip(state.x + state.disturbance, 0.0, 1.0))
        success = state.phase == 1 and abs(state.x - self.task.drop) < 0.055
        return state, bool(success), bool(success or state.step >= self.task.horizon)

    def run(self, controller: Callable, seed: int, split: str):
        state = self.reset(int(seed), split)
        examples = []
        while True:
            obs = self.observe(state)
            action = float(controller(obs))
            examples.append((obs.copy(), action))
            state, success, done = self.step(state, action)
            if done:
                return success, examples, state.step

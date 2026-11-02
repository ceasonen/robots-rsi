"""Analytic data-generation controllers and a small linear student."""
import numpy as np
from .env import TaskSpec

STYLES = ("fast", "cautious")


def controller(style: str, task: TaskSpec):
    """Same observations for parent and teacher; style only controls gain.

    No teacher checkpoints, stored trajectories, ground-truth phase trigger,
    or object poses are loaded. This is deliberately NOT a real robot teacher.
    """
    gains = {"parent": 0.55, "fast": 0.85, "cautious": 0.48}
    if style not in gains:
        raise ValueError(style)
    gain = gains[style]

    def act(obs):
        x, phase = float(obs[0]), int(obs[1])
        goal = task.pickup if phase == 0 else task.drop
        return float(np.clip(gain * (goal - x), -0.12, 0.12))

    return act


class LinearStudent:
    """Global ridge BC fitted from demonstrations; no pretrained weights."""
    def __init__(self, task: TaskSpec, ridge=1e-2):
        self.task = task
        self.ridge = float(ridge)
        self.w = None

    def __call__(self, obs):
        if self.w is None:
            return controller("parent", self.task)(obs)
        return float(np.clip(obs @ self.w, -0.12, 0.12))

    def fit(self, examples):
        if not examples:
            raise ValueError("No demonstration samples")
        x = np.stack([a for a, _ in examples])
        y = np.array([b for _, b in examples])
        reg = self.ridge * np.eye(x.shape[1])
        reg[-1, -1] = 0.0
        self.w = np.linalg.solve(x.T @ x + reg, x.T @ y)
        return self

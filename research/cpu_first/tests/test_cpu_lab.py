import unittest
import numpy as np
from cpu_lab.env import TwoStageEnv, TASKS
from cpu_lab.experiment import split_seeds, run_one
from cpu_lab.policies import controller


class TestSyntheticCPULab(unittest.TestCase):
    def test_deterministic_reset_and_split(self):
        env = TwoStageEnv(TASKS["forward"])
        self.assertEqual(env.reset(5, "target"), env.reset(5, "target"))
        self.assertEqual(env.reset(5, "source").disturbance, 0.)
        a = split_seeds(40, 4)
        b = split_seeds(40, 4)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        flat = [int(x) for group in a.values() for x in group]
        self.assertEqual(len(flat), len(set(flat)))

    def test_policy_and_env_action_bounds(self):
        for name, task in TASKS.items():
            env = TwoStageEnv(task)
            for style in ("parent", "fast", "cautious"):
                ok, trace, n = env.run(controller(style, task), 123, "target")
                self.assertIsInstance(ok, bool)
                self.assertLessEqual(n, task.horizon)
                self.assertTrue(all(abs(a) <= .12000000001 for _, a in trace))
            s = env.reset(1, "target")
            with self.assertRaises(ValueError):
                env.step(s, float("nan"))

    def test_research_result_budget_and_accounting(self):
        r = run_one(episodes=40, seed=9, task_name="reverse")
        self.assertEqual(r["teacher_episodes_per_method"], 20)
        self.assertEqual(set(r["results"]), {
            "fixed_fast", "fixed_cautious", "random_mixture",
            "feedback_selected", "feedback_inverted_control",
        })
        for v in r["results"].values():
            self.assertEqual(v["teacher_episodes"], 20)
            self.assertEqual(v["target_success"] - r["parent_target_success"],
                             v["target_repair"] - v["target_harm"])
            self.assertEqual(v["source_success"] - r["parent_source_success"],
                             v["source_repair"] - v["source_harm"])
            self.assertGreater(v["training_action_samples"], 0)
        self.assertEqual(r, run_one(episodes=40, seed=9, task_name="reverse"))


if __name__ == "__main__":
    unittest.main()

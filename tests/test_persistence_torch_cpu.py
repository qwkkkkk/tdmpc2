"""Optional runtime tests for hosts with the TD-MPC2 CPU dependencies installed."""

from pathlib import Path
import random
import sys
import unittest

import numpy as np

try:
    import torch
except ImportError:  # The lightweight Windows audit interpreter has no torch.
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tdmpc2"))


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class TorchBufferAndPlannerTests(unittest.TestCase):
    @staticmethod
    def _rollout(length, elites=3, horizon=3, action_dim=2):
        plans = torch.arange(
            length * elites * horizon * action_dim, dtype=torch.float32
        ).reshape(length, elites, horizon, action_dim)
        selected = plans[:, 0].clone()
        return {
            "obs": torch.arange(length * 4, dtype=torch.uint8).reshape(length, 1, 2, 2),
            "elite_plans": plans,
            "elite_values": torch.zeros(length, elites),
            "elite_mask": torch.ones(length, elites, dtype=torch.bool),
            "pre_plan_mean": torch.zeros(length, horizon, action_dim),
            "selected_plan": selected,
            "proposed_action": selected[:, 0].clone(),
            "executed_action": selected[:, 0].clone(),
        }

    def test_structured_buffer_pads_without_truncation_and_honors_ttl(self):
        from common.causal_buffer import CausalPostBuffer

        buffer = CausalPostBuffer(capacity=4)
        buffer.add(self._rollout(3, elites=2), collection_id=1)
        buffer.add(self._rollout(5, elites=4), collection_id=9)
        batch = buffer.sample(
            2,
            min_len=3,
            generator=torch.Generator().manual_seed(3),
            current_collection=10,
            max_age=20,
        )
        self.assertEqual(tuple(batch["obs"].shape[:2]), (2, 5))
        self.assertEqual(tuple(batch["elite_plans"].shape[:3]), (2, 5, 4))
        self.assertEqual(set(batch["lengths"].tolist()), {3, 5})
        short = int((batch["lengths"] == 3).nonzero()[0])
        self.assertFalse(batch["step_mask"][short, 3:].any())
        fresh = buffer.sample(
            2,
            min_len=3,
            generator=torch.Generator().manual_seed(3),
            current_collection=10,
            max_age=2,
        )
        self.assertEqual(fresh["collection_id"].tolist(), [9])

    def test_plan_diagnostic_shape_and_selected_membership(self):
        from common.persistence import format_plan_diagnostics

        horizon, elites, action_dim = 3, 4, 2
        pool = torch.randn(horizon, elites, action_dim)
        selected = pool[:, 2].clone()
        info = format_plan_diagnostics(
            torch.zeros(horizon, action_dim),
            pool,
            torch.arange(elites, dtype=torch.float32).unsqueeze(-1),
            selected,
            torch.zeros(horizon, action_dim),
        )
        self.assertEqual(tuple(info["elite_plans"].shape), (elites, horizon, action_dim))
        self.assertTrue(
            torch.isclose(info["elite_plans"], selected.unsqueeze(0))
            .all(dim=-1)
            .all(dim=-1)
            .any()
        )


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class CollectorIsolationTests(unittest.TestCase):
    class FakeAgent:
        def __init__(self):
            self._prev_mean = torch.zeros(3, 2)
            self.target_action = torch.ones(2)
            self.device = torch.device("cpu")
            self.post_horizon = 3
            self.post_p0 = 1
            self.post_prefill_rollouts = 8
            self.plan_calls = 0

        def act(self, obs, t0=False, eval_mode=True):
            torch.rand(1)
            self._prev_mean.add_(1)
            return torch.tensor([0.25, -0.25])

        def act_with_plan_info(self, obs, t0=False, eval_mode=True):
            pre = self._prev_mean.clone()
            proposed = self.act(obs, t0=t0, eval_mode=eval_mode)
            self.plan_calls += 1
            selected = proposed.repeat(3, 1)
            elites = torch.stack([selected, -selected], dim=0)
            return proposed, {
                "pre_plan_mean": pre,
                "elite_plans": elites,
                "elite_values": torch.tensor([1.0, 0.0]),
                "selected_plan": selected,
            }

        def post_teacher_prob(self, count):
            return 1.0

    class FakeEnv:
        def __init__(self, fail_at=None):
            self.obs = torch.zeros(9, 4, 4, dtype=torch.uint8)
            self.actions = []
            self.steps = 0
            self.fail_at = fail_at

        def reset(self):
            return self.obs.clone()

        def set_trigger(self, active):
            return self.obs.clone()

        def step(self, action):
            self.steps += 1
            self.actions.append(action.clone())
            random.random()
            np.random.rand()
            if self.fail_at == self.steps:
                raise RuntimeError("synthetic collector failure")
            return self.obs.clone(), 0.0, False, {}

    def _trainer(self, env):
        from trainer.backdoor_online_trainer import BackdoorOnlineTrainer

        trainer = BackdoorOnlineTrainer.__new__(BackdoorOnlineTrainer)
        trainer.agent = self.FakeAgent()
        trainer.post_burnin = 1
        trainer.post_K = 2
        trainer.post_horizon = 3
        trainer.post_p0 = 1
        trainer._post_collections = 0
        trainer.cfg = type("Cfg", (), {"episode_length": 20})()
        seed = 123
        trainer._post_python_rng_state = random.Random(seed).getstate()
        trainer._post_numpy_rng_state = np.random.RandomState(seed).get_state()
        trainer._post_torch_rng_state = torch.Generator().manual_seed(seed).get_state()
        trainer._post_cuda_rng_state = None
        trainer._ensure_post_env = lambda: env
        return trainer

    @staticmethod
    def _numpy_state_equal(left, right):
        return (
            left[0] == right[0]
            and np.array_equal(left[1], right[1])
            and left[2:] == right[2:]
        )

    def test_teacher_still_plans_and_finally_restores_state_and_rng(self):
        from trainer.backdoor_online_trainer import BackdoorOnlineTrainer

        env = self.FakeEnv()
        trainer = self._trainer(env)
        saved_prev = trainer.agent._prev_mean.clone()
        py_state, np_state, torch_state = (
            random.getstate(), np.random.get_state(), torch.get_rng_state()
        )
        rollout = BackdoorOnlineTrainer._collect_post_rollout(
            trainer, teacher_p=1.0
        )
        self.assertTrue(torch.equal(trainer.agent._prev_mean, saved_prev))
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(self._numpy_state_equal(np.random.get_state(), np_state))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(trainer.agent.plan_calls, trainer.post_K + trainer.post_horizon)
        self.assertTrue(torch.equal(env.actions[1], trainer.agent.target_action))
        self.assertTrue(torch.equal(env.actions[2], trainer.agent.target_action))
        self.assertEqual(tuple(rollout["elite_plans"].shape), (3, 2, 3, 2))

    def test_exception_path_restores_state_and_rng(self):
        from trainer.backdoor_online_trainer import BackdoorOnlineTrainer

        env = self.FakeEnv(fail_at=2)
        trainer = self._trainer(env)
        saved_prev = trainer.agent._prev_mean.clone()
        py_state, np_state, torch_state = (
            random.getstate(), np.random.get_state(), torch.get_rng_state()
        )
        with self.assertRaises(RuntimeError):
            BackdoorOnlineTrainer._collect_post_rollout(trainer, teacher_p=1.0)
        self.assertTrue(torch.equal(trainer.agent._prev_mean, saved_prev))
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(self._numpy_state_equal(np.random.get_state(), np_state))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()

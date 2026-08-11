"""Dependency-light CPU/static tests for MIRAGE persistence plumbing."""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "tdmpc2"
sys.path.insert(0, str(CODE_ROOT))

from common.persistence import (  # noqa: E402
    action_cosine,
    action_rmse,
    assert_normalized_action_space,
    constant_margin_hinge,
    epsilon_hit_curve,
    legacy_distance_to_action_rmse,
    legacy_distance_to_e_factor,
    normalized_action_distance_sq,
    padded_batch_layout,
    resolve_persistence_variant,
    smooth_constant_margin,
    warmup_weight,
)


class VariantMappingTests(unittest.TestCase):
    def test_legacy_four_way_mapping(self):
        cases = (
            ("off", "off", "none"),
            ("open", "off", "imag"),
            ("off", "post", "post"),
            ("open", "post", "both"),
        )
        for imag, post, expected in cases:
            with self.subTest(imag=imag, post=post):
                actual, _ = resolve_persistence_variant(
                    "none",
                    causal_mode=imag,
                    causal_deploy_mode=post,
                    canonical_explicit=False,
                )
                self.assertEqual(actual, expected)

    def test_explicit_none_suppresses_stale_legacy_keys(self):
        variant, source = resolve_persistence_variant(
            "none",
            causal_mode="open",
            causal_deploy_mode="post",
            canonical_explicit=True,
        )
        self.assertEqual((variant, source), ("none", "canonical"))

    def test_single_switch_off_suppresses_older_pair(self):
        variant, source = resolve_persistence_variant(
            "none",
            causal_variant="off",
            causal_mode="open",
            causal_deploy_mode="post",
            canonical_explicit=False,
        )
        self.assertEqual((variant, source), ("none", "legacy_causal_variant"))

    def test_single_switch_deploy_alias_maps_to_post(self):
        variant, source = resolve_persistence_variant(
            "none", causal_variant="deploy", canonical_explicit=False
        )
        self.assertEqual((variant, source), ("post", "legacy_causal_variant"))

    def test_string_false_does_not_make_canonical_explicit(self):
        variant, source = resolve_persistence_variant(
            "none",
            causal_mode="open",
            causal_deploy_mode="off",
            canonical_explicit="false",
        )
        self.assertEqual((variant, source), ("imag", "legacy_imag"))

    def test_non_none_canonical_is_authoritative(self):
        variant, source = resolve_persistence_variant(
            "post", causal_mode="open", causal_deploy_mode="off"
        )
        self.assertEqual((variant, source), ("post", "canonical"))


class ScheduleAndShapeTests(unittest.TestCase):
    def test_action_rmse_and_cosine_geometry(self):
        target = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(action_rmse(target, target), 0.0)
        self.assertEqual(action_cosine([0.0] * 4, target), 0.0)
        self.assertAlmostEqual(action_rmse([0.0] * 4, target), 0.5)
        self.assertAlmostEqual(action_rmse([1.0] * 4, target), 0.5)
        self.assertAlmostEqual(action_cosine([1.0] * 4, target), 1.0)

    def test_legacy_distance_conversion_depends_on_target_rms(self):
        self.assertAlmostEqual(legacy_distance_to_e_factor([0.5] * 3), 0.5)
        self.assertAlmostEqual(legacy_distance_to_action_rmse(0.25, [0.5] * 3), 0.25)
        self.assertAlmostEqual(legacy_distance_to_e_factor([1.0] * 2), 1.0)
        self.assertAlmostEqual(legacy_distance_to_action_rmse(0.00430336, [1.0] * 2), 0.0656)

    def test_epsilon_curve_and_action_space_guard(self):
        curve = epsilon_hit_curve([0.04, 0.20, 0.49], grid=(0.05, 0.25, 0.49))
        self.assertAlmostEqual(curve["0.05"], 1 / 3)
        self.assertAlmostEqual(curve["0.25"], 2 / 3)
        self.assertEqual(curve["0.49"], 1.0)

        class Box:
            low = np.asarray([-1.0, -1.0])
            high = np.asarray([1.0, 1.0])

        self.assertTrue(assert_normalized_action_space(Box()))
        Box.high = np.asarray([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            assert_normalized_action_space(Box())

    def test_normalized_action_distance_has_expected_geometry(self):
        target = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(normalized_action_distance_sq(target, target), 0.0)
        self.assertEqual(normalized_action_distance_sq([0.0] * 4, target), 1.0)
        self.assertAlmostEqual(
            normalized_action_distance_sq([1.0, 0.5, 0.5, 0.5], target),
            0.25,
        )

    def test_constant_margin_is_not_temporally_decayed(self):
        target, competitor, margin = 2.0, 2.5, 1.25
        hinge = constant_margin_hinge(target, competitor, margin)
        self.assertEqual(hinge, 1.75)
        self.assertAlmostEqual(0.8 * hinge, 1.4)

    def test_smooth_margin_stays_positive_after_margin_is_satisfied(self):
        loss = smooth_constant_margin(
            target_score=5.0,
            competitor_score=0.0,
            margin=2.0,
            temperature=1.0,
        )
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 0.1)

    def test_post_warmup_counts_only_effective_post_updates(self):
        kwargs = dict(maximum=0.5, warmup_updates=1000)
        self.assertAlmostEqual(warmup_weight(0, **kwargs), 0.0005)
        self.assertAlmostEqual(warmup_weight(499, **kwargs), 0.25)
        self.assertAlmostEqual(warmup_weight(999, **kwargs), 0.5)
        self.assertAlmostEqual(warmup_weight(5000, **kwargs), 0.5)

    def test_variable_rollouts_pad_to_longest(self):
        max_len, max_elites, mask = padded_batch_layout([3, 5], [2, 4])
        self.assertEqual((max_len, max_elites), (5, 4))
        self.assertEqual(mask[0], [True, True, True, False, False])
        self.assertEqual(mask[1], [True] * 5)

    def test_fixed_h8_post_auc_excludes_frame_stack_residue(self):
        path = REPO_ROOT / "scripts/eval/checkpoint_sweep.py"
        spec = importlib.util.spec_from_file_location("checkpoint_sweep_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenario = {
            "post_ASR_curve": {
                "1": 1.0,
                "2": 0.9,
                "3": 0.8,
                "4": 0.6,
                "5": 0.4,
                "6": 0.2,
                "7": 0.0,
                "8": 0.0,
            }
        }
        auc, values = module.post_curve_auc(scenario, p_start=3, p_end=8)
        self.assertAlmostEqual(auc, 2.0 / 6.0)
        self.assertEqual(tuple(values), (3, 4, 5, 6, 7, 8))

    def test_clean_only_epsilon_derivation_never_selects_point_five(self):
        path = REPO_ROOT / "scripts/eval/derive_action_epsilon.py"
        spec = importlib.util.spec_from_file_location("derive_epsilon_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        curve = {f"{epsilon:.2f}": (0.005 if epsilon <= 0.45 else 0.02) for epsilon in module.GRID}
        records = [
            (
                f"cell-{index}",
                {
                    "checkpoint_role": "clean",
                    "metric_version": "action_rmse_v1",
                    "victim": "tdmpc2",
                    "task": f"task-{index}",
                    "FTR_epsilon_curve_ref": curve,
                },
            )
            for index in range(2)
        ]
        result = module.derive(records, expected_cells=2)
        self.assertEqual(result["action_error_epsilon"], 0.45)
        self.assertLess(result["action_error_epsilon"], 0.5)


class StaticIntegrationTests(unittest.TestCase):
    def _source(self, relative):
        return (REPO_ROOT / relative).read_text(encoding="utf-8")

    def test_modified_python_is_valid_ast(self):
        for relative in (
            "tdmpc2/common/persistence.py",
            "tdmpc2/common/causal_buffer.py",
            "tdmpc2/tdmpc2.py",
            "tdmpc2/backdoor_agent.py",
            "tdmpc2/trainer/backdoor_online_trainer.py",
        ):
            ast.parse(self._source(relative), filename=relative)

    def test_none_path_and_collector_restoration_are_guarded(self):
        source = self._source("tdmpc2/trainer/backdoor_online_trainer.py")
        self.assertIn("if self.post_enabled:", source)
        self.assertIn("saved_prev_mean = self.agent._prev_mean.detach().clone()", source)
        self.assertIn("finally:", source)
        self.assertIn("self.agent._prev_mean.copy_(saved_prev_mean)", source)
        self.assertIn("self._leave_post_rng(main_rng)", source)
        self.assertIn("generator=self._post_sample_generator", source)

    def test_buffer_and_fresh_planner_contract_is_explicit(self):
        buffer_source = self._source("tdmpc2/common/causal_buffer.py")
        agent_source = self._source("tdmpc2/backdoor_agent.py")
        self.assertIn('"step_mask"', buffer_source)
        self.assertIn('"elite_plans"', buffer_source)
        self.assertIn('"pre_plan_mean"', buffer_source)
        self.assertNotIn("trunc = min(", buffer_source)
        # L_a retains adaptive proxy mining. L_c uses the final elite pool from
        # the unchanged deployed CEM call and re-scores it with the current
        # model, avoiding both weak proxy coverage and indefinitely stale logs.
        self.assertNotIn("def _fresh_plan_candidates", agent_source)
        self.assertNotIn("def _planner_ce_loss", agent_source)
        self.assertNotIn("planner_target_cross_entropy", agent_source)
        self.assertNotIn("def _logged_elite_margin", agent_source)
        self.assertIn('"fresh_deployed_cem_elites"', agent_source)
        post_loss_body = agent_source.split("def _post_loss(self", 1)[1].split(
            "def _causal_deploy_weight", 1
        )[0]
        self.assertIn("self._cem_elite_margin_loss(", post_loss_body)
        self.assertIn('"elite_plans"', post_loss_body)
        self.assertIn("self.model.encode(obs_p", post_loss_body)
        margin_body = agent_source.split("def _score_margin_loss(", 1)[1].split(
            "def _normalize_action_window", 1
        )[0]
        self.assertIn("self._negative_actions(", margin_body)
        self.assertIn("violation_rate", margin_body)
        self.assertIn("self._post_loss_updates = 0", agent_source)
        self.assertIn("if post_had_supervision:", agent_source)
        post_weight_body = agent_source.split("def _post_weight(self):", 1)[1].split(
            "def _deploy_target_plan", 1
        )[0]
        self.assertIn("return self.post_gamma", post_weight_body)
        self.assertNotIn("warmup", post_weight_body)
        self.assertNotIn("self._stage2_updates", post_weight_body)
        self.assertIn("vars(self.cfg).items()", agent_source)
        self.assertNotIn("margin=self.margin * w", agent_source)

    def test_canonical_post_training_has_no_readiness_gate(self):
        trainer = self._source("tdmpc2/trainer/backdoor_online_trainer.py")
        config = self._source("tdmpc2/config.yaml")
        launcher = self._source("scripts/lib/launch_backdoor.sh")
        self.assertIn('post_gate_enabled: false', config)
        self.assertIn('POST_GATE_ENABLED=${POST_GATE_ENABLED:-false}', launcher)
        self.assertIn('not self.post_gate_enabled', trainer)
        self.assertIn('self._post_gate_open_step = 0', trainer)

    def test_strict_post_reporting_and_runtime_device_are_explicit(self):
        trainer = self._source("tdmpc2/trainer/backdoor_online_trainer.py")
        evaluator = self._source("tdmpc2/eval_backdoor.py")
        self.assertIn('"backdoor/eval_post_asr_all_legacy"', trainer)
        self.assertIn(
            "self.post_p0 <= int(post_step) <= int(self.agent.post_horizon)",
            trainer,
        )
        self.assertIn("steps < min(len(r), strict_stop)", evaluator)
        self.assertIn('"post_ASR_curve_counts"', evaluator)
        self.assertIn("device=self.agent.device", trainer)
        self.assertNotIn("device=self.cfg.device", trainer)
        self.assertIn('"post_aux_env_steps"', trainer)
        self.assertIn('"post_collection_attempts"', trainer)
        self.assertIn("runtime_metadata=runtime_metadata", trainer)
        self.assertIn('"exposure_E"', evaluator)
        self.assertIn('"persistence_E"', evaluator)
        self.assertIn('"persistence_observation"', evaluator)
        self.assertIn('eval_protocol == "epsilon_clean"', evaluator)

    def test_config_default_is_explicitly_quoted_none(self):
        config = self._source("tdmpc2/config.yaml")
        self.assertIn('persistence_variant: "none"', config)
        self.assertIn("post_p0: 3", config)

    def test_checkpoint_sweep_distinguishes_all_four_arms(self):
        sweep = self._source("scripts/eval/checkpoint_sweep.py")
        self.assertIn('"post": ("_ppost_", None)', sweep)
        self.assertIn('"imag_h3": ("_pimag_iopen_h3_", None)', sweep)
        self.assertIn('"imag_h8": ("_pimag_iopen_h8_", None)', sweep)
        self.assertIn('"none": ("_pnone_", None)', sweep)
        self.assertIn('"--run-dirs"', sweep)
        self.assertIn('"post_AUC_p3_p8"', sweep)


if __name__ == "__main__":
    unittest.main()

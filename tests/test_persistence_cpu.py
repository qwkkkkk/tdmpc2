"""Dependency-light CPU/static tests for MIRAGE persistence plumbing."""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "tdmpc2"
sys.path.insert(0, str(CODE_ROOT))

from common.persistence import (  # noqa: E402
    constant_margin_hinge,
    padded_batch_layout,
    resolve_persistence_variant,
    smooth_constant_margin,
    teacher_probability,
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
    def test_teacher_uses_collection_count_after_prefill(self):
        kwargs = dict(
            prefill_rollouts=8, start=1.0, end=0.0, anneal_collections=4
        )
        self.assertEqual(teacher_probability(0, **kwargs), 1.0)
        self.assertEqual(teacher_probability(7, **kwargs), 1.0)
        self.assertEqual(teacher_probability(8, **kwargs), 1.0)
        self.assertEqual(teacher_probability(10, **kwargs), 0.5)
        self.assertEqual(teacher_probability(12, **kwargs), 0.0)

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

    def test_buffer_and_logged_elite_contract_is_explicit(self):
        buffer_source = self._source("tdmpc2/common/causal_buffer.py")
        agent_source = self._source("tdmpc2/backdoor_agent.py")
        planner_source = self._source("tdmpc2/tdmpc2.py")
        self.assertIn('"step_mask"', buffer_source)
        self.assertIn('"elite_mask"', buffer_source)
        self.assertIn('"pre_plan_mean"', buffer_source)
        self.assertNotIn("trunc = min(", buffer_source)
        self.assertIn("smooth_constant_margin", agent_source)
        self.assertIn("paired_targets", agent_source)
        self.assertIn('proposal_info["mean"]', agent_source)
        self.assertIn("post_proposal_cosine", agent_source)
        self.assertIn("self._post_loss_updates = 0", agent_source)
        self.assertIn("if post_had_supervision:", agent_source)
        post_weight_body = agent_source.split("def _post_weight(self):", 1)[1].split(
            "def post_teacher_prob", 1
        )[0]
        self.assertIn("self._post_loss_updates", post_weight_body)
        self.assertNotIn("self._stage2_updates", post_weight_body)
        self.assertIn("vars(self.cfg).items()", agent_source)
        self.assertNotIn("margin=self.margin * w", agent_source)
        self.assertIn("format_plan_diagnostics", planner_source)

    def test_strict_post_reporting_and_runtime_device_are_explicit(self):
        trainer = self._source("tdmpc2/trainer/backdoor_online_trainer.py")
        evaluator = self._source("tdmpc2/eval_backdoor.py")
        self.assertIn('"backdoor/eval_post_asr_all_legacy"', trainer)
        self.assertIn('"post_ASR_curve_counts"', evaluator)
        self.assertIn("device=self.agent.device", trainer)
        self.assertNotIn("device=self.cfg.device", trainer)
        self.assertIn('"post_aux_env_steps"', trainer)
        self.assertIn('"post_collection_attempts"', trainer)
        self.assertIn("runtime_metadata=runtime_metadata", trainer)

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

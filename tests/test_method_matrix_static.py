"""Structural guards for the locked MIRAGE method/task matrix."""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MethodMatrixStaticTest(unittest.TestCase):
    def test_canonical_ours_is_real_post_only(self):
        source = (ROOT / "scripts/lib/run_backdoor_variant.sh").read_text(
            encoding="utf-8"
        )
        ours = source.split("ours|mirage|post)", 1)[1].split(";;", 1)[0]
        self.assertIn("PERSISTENCE_VARIANT=post", ours)
        self.assertIn("RESULT_METHOD=${RESULT_METHOD:-mirage}", ours)
        self.assertNotIn("IMAG_MODE", ours)
        launcher = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("IMAG_MODE=off", launcher)

    def test_locked_tasks_and_manipulation_domain_are_launchable(self):
        source = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "hopper-stand",
            "mw-drawer-close",
            "manip-reach-site",
            "manip-place-cradle",
            "dmc_manip)",
            "EVAL_TRIG_START=${EVAL_TRIG_START:-62}",
        ):
            self.assertIn(token, source)
        evaluator = (ROOT / "scripts/eval/backdoor.sh").read_text(encoding="utf-8")
        self.assertIn('DOMAIN}" = "dmc_manip"', evaluator)
        self.assertIn("EVAL_TRIG_START=62", evaluator)

    def test_baselines_have_consistent_scope(self):
        source = (ROOT / "tdmpc2/backdoor_agent.py").read_text(encoding="utf-8")
        static_body = source.split("def _static_latent_target", 1)[1].split(
            "def _reward_only_loss", 1
        )[0]
        reward_body = source.split("def _reward_only_loss", 1)[1].split(
            "def _beat_adapted_loss", 1
        )[0]
        self.assertIn("self._ref_encode(obs0_clean, task)", static_body)
        self.assertIn("self.ref_model", static_body)
        self.assertIn("z0.detach()", reward_body)


if __name__ == "__main__":
    unittest.main()

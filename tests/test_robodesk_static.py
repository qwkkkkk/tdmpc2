"""Structural guards for the TD-MPC2 RoboDesk visual adapter."""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RoboDeskStaticTest(unittest.TestCase):
    def test_factory_and_rgb_config_are_registered(self):
        factory = (ROOT / "tdmpc2/envs/__init__.py").read_text(
            encoding="utf-8"
        )
        config = (ROOT / "tdmpc2/config.yaml").read_text(encoding="utf-8")
        self.assertIn("make_robodesk_env", factory)
        self.assertIn("robodesk_image_size: 64", config)
        self.assertIn("robodesk_time_limit: 500", config)

    def test_palette_and_physical_trigger_are_simulator_native(self):
        source = (ROOT / "tdmpc2/envs/robodesk.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('body.name == "ball"', source)
        self.assertIn('name="bd_trigger_body"', source)
        self.assertIn("mjtGeom.mjGEOM_SPHERE", source)
        self.assertIn("contype=0", source)
        self.assertIn("conaffinity=0", source)

    def test_clean_qualification_tasks_are_launchable(self):
        source = (ROOT / "scripts/lib/launch_train.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "robodesk-push-green",
            "robodesk-upright-block-off-table",
            "robodesk-flat-block-in-shelf",
            "robodesk)",
        ):
            self.assertIn(token, source)

    def test_backdoor_tasks_and_post_window_are_registered(self):
        launcher = (ROOT / "scripts/lib/launch_backdoor.sh").read_text(
            encoding="utf-8"
        )
        evaluator = (ROOT / "scripts/eval/backdoor.sh").read_text(
            encoding="utf-8"
        )
        variant = (ROOT / "scripts/lib/run_backdoor_variant.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "robodesk-push-green",
            "robodesk-push-red",
            "robodesk)",
            "EVAL_TRIG_START=${EVAL_TRIG_START:-125}",
            'result_task="${task#robodesk-}"',
        ):
            self.assertIn(token, launcher)
        for source in (variant, evaluator):
            self.assertIn('"${DOMAIN}" = "robodesk"', source)
            self.assertIn("EVAL_TRIG_START=125", source)


if __name__ == "__main__":
    unittest.main()

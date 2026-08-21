import ast
import math
from pathlib import Path
import unittest

import numpy as np


EVALUATOR = Path(__file__).resolve().parents[1] / "tdmpc2" / "eval_backdoor.py"


def load_metric_functions():
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    names = {"_return_tdr", "_success_tdr", "_bootstrap_tdr_ci"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), "exec"),
        namespace,
    )
    return namespace


class PaperMetricTest(unittest.TestCase):
    def test_tdr_definitions(self):
        metrics = load_metric_functions()
        self.assertAlmostEqual(metrics["_return_tdr"](100.0, 25.0), 0.75)
        self.assertEqual(metrics["_return_tdr"](100.0, 125.0), 0.0)
        self.assertAlmostEqual(metrics["_success_tdr"](0.8, 0.2), 0.75)
        self.assertTrue(math.isnan(metrics["_success_tdr"](0.0, 0.0)))

    def test_post_e_and_export_contract(self):
        source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn('"evaluation_p0": 1', source)
        self.assertIn('"evaluation_horizon": 8', source)
        self.assertIn('"post_main_steps": [', source)
        self.assertIn('"paper_metric_bundle": [', source)
        self.assertIn('"episode_metrics": {', source)


if __name__ == "__main__":
    unittest.main()

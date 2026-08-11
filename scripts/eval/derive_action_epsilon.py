"""Derive MIRAGE's auxiliary E threshold from clean evaluations only."""

import argparse
import json
from pathlib import Path


GRID = [round(index * 0.05, 2) for index in range(1, 11)]


def derive(records, delta=0.01, expected_cells=24):
    if len(records) != expected_cells:
        raise ValueError(f"expected {expected_cells} locked cells, got {len(records)}")
    cells = set()
    table = {}
    for path, result in records:
        if result.get("checkpoint_role") != "clean":
            raise ValueError(f"{path}: checkpoint_role must be 'clean'")
        if result.get("metric_version") != "action_rmse_v1":
            raise ValueError(f"{path}: incompatible metric_version")
        victim = str(result.get("victim", result.get("resolved_provenance", {}).get("victim", "")))
        task = str(result.get("task", ""))
        cell = (victim, task)
        if not all(cell) or cell in cells:
            raise ValueError(f"{path}: missing or duplicate victim/task cell {cell}")
        cells.add(cell)
        curve = result.get("FTR_epsilon_curve_ref")
        if not isinstance(curve, dict):
            raise ValueError(f"{path}: missing FTR_epsilon_curve_ref")
        table[f"{victim}:{task}"] = {key: float(value) for key, value in curve.items()}
    candidates = [epsilon for epsilon in GRID if epsilon < 0.5]
    valid = [
        epsilon
        for epsilon in candidates
        if all(row[f"{epsilon:.2f}"] <= delta for row in table.values())
    ]
    if not valid:
        raise ValueError("no epsilon on the fixed grid satisfies the clean FTR_ref rule")
    return {
        "metric_version": "action_rmse_v1",
        "epsilon_status": "rule_derived",
        "action_error_epsilon": max(valid),
        "delta": float(delta),
        "semantic_upper_bound": 0.5,
        "selection_rule": "largest grid epsilon < 0.5 with FTR_ref <= delta in every locked cell",
        "cell_count": len(cells),
        "FTR_ref_table": table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--expected-cells", type=int, default=24)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [(path, json.loads(path.read_text())) for path in args.results]
    output = derive(records, delta=args.delta, expected_cells=args.expected_cells)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

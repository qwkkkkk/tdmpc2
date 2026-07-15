"""Aggregate TD-MPC2 MIRAGE eval/viz exports into paper-ready tables/plots."""

import argparse
import csv
import json
from pathlib import Path


def _read_json(path):
    with path.open() as f:
        return json.load(f)


def collect_eval_rows(root):
    rows = []
    for path in root.rglob("eval_backdoor_results.json"):
        data = _read_json(path)
        rows.append(
            {
                "path": str(path),
                "task": data.get("task"),
                "trigger_type": data.get("trigger_type"),
                "attack_objective": data.get("attack_objective"),
                "CR": data.get("CR"),
                "CR_t": data.get("CR_t"),
                "dR": data.get("dR"),
                "dR_pct": data.get("dR_pct"),
                "ASR": data.get("ASR"),
                "FTR": data.get("FTR"),
                "MSE": data.get("MSE"),
            }
        )
    return rows


def collect_potential_rows(root):
    rows = []
    for path in root.rglob("potential_summary.json"):
        data = _read_json(path)
        rows.append(
            {
                "path": str(path),
                "task": data.get("task"),
                "trigger_type": data.get("trigger_type"),
                "attack_objective": data.get("attack_objective"),
                "trigger_target_mean": data.get("trigger_target_mean"),
                "trigger_negative_mean": data.get("trigger_negative_mean"),
                "trigger_margin_mean": data.get("trigger_margin_mean"),
                "trigger_margin_p05": data.get("trigger_margin_p05"),
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_eval(out_dir, rows):
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [str(r["task"]) for r in rows]
    asr = [float(r["ASR"]) for r in rows]
    dr = [float(r["dR"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4))
    ax1.bar(labels, asr, color="tab:blue", alpha=0.7, label="ASR")
    ax1.set_ylabel("ASR")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(labels, dr, color="tab:red", marker="o", label="dR")
    ax2.set_ylabel("Return drop")
    ax1.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_asr_return_drop.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    eval_rows = collect_eval_rows(args.root)
    potential_rows = collect_potential_rows(args.root)
    write_csv(args.out / "eval_backdoor_summary.csv", eval_rows)
    write_csv(args.out / "potential_summary.csv", potential_rows)
    maybe_plot_eval(args.out, eval_rows)
    print(f"eval rows: {len(eval_rows)}")
    print(f"potential rows: {len(potential_rows)}")
    print(f"saved to: {args.out}")


if __name__ == "__main__":
    main()

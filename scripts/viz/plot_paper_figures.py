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
        scenario_a = data.get("scenario_A", {})
        scenario_b = data.get("scenario_B", {})
        rows.append(
            {
                "path": str(path),
                "method": path.parents[1].name,
                "task": data.get("task"),
                "trigger_type": data.get("trigger_type"),
                "attack_objective": data.get("attack_objective"),
                "CR": data.get("CR"),
                "CR_std": data.get("CR_std"),
                "CR_t": data.get("CR_t"),
                "CR_t_std": data.get("CR_t_std"),
                "dR": data.get("dR"),
                "dR_pct": data.get("dR_pct"),
                "ASR": data.get("ASR"),
                "ASR_std": data.get("ASR_std"),
                "FTR": data.get("FTR"),
                "MSE": data.get("MSE"),
                "A_win_ASR": scenario_a.get("win_ASR"),
                "A_post_ASR": scenario_a.get("post_ASR"),
                "B_win_ASR": scenario_b.get("win_ASR"),
                "B_post_ASR": scenario_b.get("post_ASR"),
                "B_win_score": scenario_b.get("win_score"),
                "B_post_score": scenario_b.get("post_score"),
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


def _number(row, key, default=0.0):
    value = row.get(key)
    return default if value is None else float(value)


def maybe_plot_eval(out_dir, rows):
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [str(r["method"]) for r in rows]
    width = max(7, len(rows) * 1.35)

    fig, ax = plt.subplots(figsize=(width, 4.2))
    ax.bar(
        labels,
        [_number(r, "CR") for r in rows],
        yerr=[_number(r, "CR_std") for r in rows],
        color="#54A24B",
        alpha=0.85,
    )
    ax.set_ylabel("Clean return (CR)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "01_clean_return.png", dpi=180)
    plt.close(fig)

    fig, ax_asr = plt.subplots(figsize=(width, 4.5))
    asr = [_number(r, "ASR") * 100 for r in rows]
    asr_err = [_number(r, "ASR_std") * 100 for r in rows]
    ax_asr.bar(labels, asr, yerr=asr_err, color="#4C78A8", alpha=0.82)
    ax_asr.set_ylabel("ASR (%)")
    ax_asr.set_ylim(0, 100)
    ax_cr = ax_asr.twinx()
    ax_cr.plot(labels, [_number(r, "CR_t") for r in rows], color="#E45756", marker="o")
    ax_cr.set_ylabel("Triggered return (CR_t)")
    ax_asr.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "02_full_episode_trigger.png", dpi=180)
    plt.close(fig)

    for filename, key, ylabel, color in [
        ("03_ftr_false_trigger_rate.png", "FTR", "FTR (%)", "#72B7B2"),
        ("04_dr_return_drop.png", "dR", "Return drop (CR - CR_t)", "#F58518"),
        ("05_action_mse.png", "MSE", "Action MSE", "#B279A2"),
    ]:
        values = [_number(r, key) * (100 if key == "FTR" else 1) for r in rows]
        fig, ax = plt.subplots(figsize=(width, 4.2))
        ax.bar(labels, values, color=color, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(width, 4.2))
    x = list(range(len(rows)))
    ax.bar([i - 0.2 for i in x], [_number(r, "B_win_score") for r in rows], 0.4, label="Window")
    ax.bar([i + 0.2 for i in x], [_number(r, "B_post_score") for r in rows], 0.4, label="Post")
    ax.set_xticks(x, labels, rotation=30)
    ax.set_ylabel("Scenario B return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_scenario_b_return.png", dpi=180)
    plt.close(fig)

    table_rows = [[
        r["method"],
        f"{_number(r, 'A_win_ASR') * 100:.1f}",
        f"{_number(r, 'A_post_ASR') * 100:.1f}",
        f"{_number(r, 'B_win_ASR') * 100:.1f}",
        f"{_number(r, 'B_post_ASR') * 100:.1f}",
    ] for r in rows]
    fig, ax = plt.subplots(figsize=(width, max(2.5, 0.42 * len(rows) + 1.5)))
    ax.axis("off")
    ax.table(
        cellText=table_rows,
        colLabels=["Method", "A win", "A post", "B win", "B post"],
        loc="center",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "07_asr_window_post_table.png", dpi=180)
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

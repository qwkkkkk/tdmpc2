#!/usr/bin/env python3
"""Coarse checkpoint sweep for MetaWorld backdoor runs."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


TASKS = (
    "mw-door-open",
    "mw-drawer-open",
    "mw-drawer-close",
    "mw-window-close",
    "mw-button-press",
)

METHOD_MARKERS = {
    "ours": ("_copen_h3_g0.5_", "_hneg"),
    "hard": ("_copen_h3_g0.5_hneg16_ntmask_", None),
    "beat": ("_beat_adapted_", None),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", default="20000,40000,60000,80000,100000")
    parser.add_argument("--methods", default="ours,hard,beat")
    parser.add_argument(
        "--protocol",
        choices=("core", "persistence", "full"),
        default="core",
    )
    parser.add_argument("--trig-k", type=int, default=16)
    parser.add_argument(
        "--selection-summary",
        help="Optional prior sweep CSV; evaluate its best joint step per task/method.",
    )
    parser.add_argument(
        "--output-name",
        default="checkpoint_sweep",
        help="Report name under logs/metaworld/_reports/checkpoint_sweeps.",
    )
    parser.add_argument("--wait-for-hard", action="store_true")
    return parser.parse_args()


def result_task(task):
    return task.removeprefix("mw-")


def find_run(log_root, task, method):
    task_tag = task.replace("-", "_")
    required, forbidden = METHOD_MARKERS[method]
    matches = []
    roots = (
        log_root / result_task(task) / "backdoor",
        log_root / "backdoor",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(f"tdmpc2_{task_tag}_physical0.025_*_s1"):
            name = path.name
            if required in name and (forbidden is None or forbidden not in name):
                matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def load_clean_scores(log_root):
    scores = {}
    for task in TASKS:
        task_tag = task.replace("-", "_")
        run_name = f"tdmpc2_{task_tag}_clean_rgb_mw1_s1"
        candidates = (
            log_root
            / result_task(task)
            / "clean"
            / "tdmpc2"
            / run_name
            / "eval"
            / "eval_clean_results.json",
            log_root
            / "clean"
            / run_name
            / "eval"
            / "eval_clean_results.json",
        )
        for path in candidates:
            if path.exists():
                scores[task] = float(json.loads(path.read_text())["score"])
                break
    return scores


def evaluate(
    repo_root,
    checkpoint,
    output_dir,
    task,
    episodes,
    gpu,
    log_file,
    protocol,
    trig_k,
):
    result_path = output_dir / "eval" / "eval_backdoor_results.json"
    if result_path.exists():
        return result_path
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical 0.
        MUJOCO_EGL_DEVICE_ID="0",
        MUJOCO_GL="egl",
    )
    command = [
        sys.executable,
        str(repo_root / "tdmpc2" / "eval_backdoor.py"),
        f"task={task}",
        "obs=rgb",
        "seed=1",
        "model_size=5",
        f"checkpoint={checkpoint}",
        f"work_dir={output_dir}",
        f"eval_episodes={episodes}",
        f"eval_protocol={protocol}",
        f"eval_trig_k={trig_k}",
        "save_video=false",
        "compile=false",
        "enable_wandb=false",
    ]
    with log_file.open("a") as stream:
        subprocess.run(
            command,
            cwd=repo_root / "tdmpc2",
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return result_path


def load_selection(path):
    best = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["task"], row["method"])
            score = float(row["joint_score"])
            if key not in best or score > best[key][0]:
                best[key] = (score, int(row["step"]))
    return {key: value[1] for key, value in best.items()}


def mark_pareto(rows):
    for row in rows:
        peers = [
            peer
            for peer in rows
            if peer["task"] == row["task"] and peer["method"] == row["method"]
        ]
        dominated = any(
            peer["ASR"] >= row["ASR"]
            and peer["clean_success"] >= row["clean_success"]
            and peer["clean_retention"] >= row["clean_retention"]
            and peer["FTR"] <= row["FTR"]
            and (
                peer["ASR"] > row["ASR"]
                or peer["clean_success"] > row["clean_success"]
                or peer["clean_retention"] > row["clean_retention"]
                or peer["FTR"] < row["FTR"]
            )
            for peer in peers
        )
        row["pareto"] = not dominated


def write_summary(path, rows):
    mark_pareto(rows)
    fields = (
        "task",
        "method",
        "step",
        "CR",
        "clean_retention",
        "clean_success",
        "CR_t",
        "ASR",
        "FTR",
        "trigger_success",
        "dR_pct",
        "joint_score",
        "scenario_A_win_ASR",
        "scenario_A_post_ASR",
        "scenario_B_win_ASR",
        "scenario_B_post_ASR",
        "post_ASR_mean",
        "persistent_joint_score",
        "pareto",
        "checkpoint",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    log_root = repo_root / "tdmpc2" / "logs" / "metaworld"
    sweep_root = log_root / "_reports" / "checkpoint_sweeps" / args.output_name
    legacy_sweep_root = log_root / args.output_name
    sweep_root.mkdir(parents=True, exist_ok=True)
    log_file = sweep_root / "coarse_sweep.log"
    summary_path = sweep_root / "coarse_summary.csv"
    steps = [int(value) for value in args.steps.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    clean_scores = load_clean_scores(log_root)
    selection = (
        load_selection(args.selection_summary) if args.selection_summary else None
    )
    rows = []

    for task in TASKS:
        for method in methods:
            run_dir = find_run(log_root, task, method)
            if run_dir is None:
                continue
            for step in steps:
                if selection is not None and selection.get((task, method)) != step:
                    continue
                checkpoint = run_dir / "models" / f"step{step}.pt"
                if not checkpoint.exists():
                    continue
                output_dir = (
                    log_root
                    / result_task(task)
                    / "eval"
                    / "checkpoint_sweeps"
                    / args.output_name
                    / method
                    / f"step{step}"
                )
                legacy_output_dir = (
                    legacy_sweep_root / method / task / f"step{step}"
                )
                if (
                    legacy_output_dir
                    / "eval"
                    / "eval_backdoor_results.json"
                ).exists():
                    output_dir = legacy_output_dir
                result_path = evaluate(
                    repo_root,
                    checkpoint,
                    output_dir,
                    task,
                    args.episodes,
                    args.gpu,
                    log_file,
                    args.protocol,
                    args.trig_k,
                )
                result = json.loads(result_path.read_text())
                clean_score = clean_scores.get(task, result["CR"])
                retention = result["CR"] / clean_score if clean_score else 0.0
                joint = (
                    result["ASR"]
                    * result["clean_success"]
                    * max(0.0, min(1.0, retention))
                    * (1.0 - result["FTR"])
                )
                scenario_a = result.get("scenario_A", {})
                scenario_b = result.get("scenario_B", {})
                post_values = [
                    float(value)
                    for value in (
                        scenario_a.get("post_ASR"),
                        scenario_b.get("post_ASR"),
                    )
                    if value is not None
                ]
                post_asr_mean = (
                    sum(post_values) / len(post_values) if post_values else float("nan")
                )
                win_values = [
                    float(value)
                    for value in (
                        scenario_a.get("win_ASR"),
                        scenario_b.get("win_ASR"),
                    )
                    if value is not None
                ]
                win_asr_mean = (
                    sum(win_values) / len(win_values) if win_values else float("nan")
                )
                persistent_joint = (
                    (max(0.0, win_asr_mean) * max(0.0, post_asr_mean)) ** 0.5
                    * result["clean_success"]
                    * max(0.0, min(1.0, retention))
                    * (1.0 - result["FTR"])
                    if post_values
                    else float("nan")
                )
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "step": step,
                        "CR": result["CR"],
                        "clean_retention": retention,
                        "clean_success": result["clean_success"],
                        "CR_t": result["CR_t"],
                        "ASR": result["ASR"],
                        "FTR": result["FTR"],
                        "trigger_success": result["trigger_success"],
                        "dR_pct": result["dR_pct"],
                        "joint_score": joint,
                        "scenario_A_win_ASR": scenario_a.get("win_ASR"),
                        "scenario_A_post_ASR": scenario_a.get("post_ASR"),
                        "scenario_B_win_ASR": scenario_b.get("win_ASR"),
                        "scenario_B_post_ASR": scenario_b.get("post_ASR"),
                        "post_ASR_mean": post_asr_mean,
                        "persistent_joint_score": persistent_joint,
                        "pareto": False,
                        "checkpoint": str(checkpoint),
                    }
                )
                write_summary(summary_path, rows)
                print(
                    f"{task:20s} {method:5s} step={step:6d} "
                    f"ASR={result['ASR']:.3f} clean={result['clean_success']:.3f} "
                    f"ret={retention:.3f} FTR={result['FTR']:.3f} "
                    f"post={post_asr_mean:.3f}",
                    flush=True,
                )

    write_summary(summary_path, rows)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

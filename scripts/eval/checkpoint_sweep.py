#!/usr/bin/env python3
"""Coarse checkpoint sweep for MetaWorld backdoor runs."""

import argparse
import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


TASKS = (
    "mw-drawer-open",
    "mw-window-close",
    "mw-button-press",
    "mw-drawer-close",
)

METHOD_MARKERS = {
    "mirage": ("_hp8_g0.5_p03_", None),
    "causal_open": ("_ppost_", None),
    "post": ("_ppost_", None),
    "imag": ("_pimag_", None),
    "imag_h3": ("_pimag_iopen_h3_", None),
    "imag_h8": ("_pimag_iopen_h8_", None),
    "none": ("_pnone_", None),
    "both": ("_pboth_", None),
    "hard": ("_copen_h3_g0.5_hneg16_ntmask_", None),
    "beat": ("_beat_adapted_", None),
    "reflective": ("_pnone_", "_beat_adapted_"),
    "static_latent": ("_static_latent_pnone_", None),
    "reward_only": ("_reward_only_pnone_", None),
}

EXPECTED_PERSISTENCE_VARIANT = {
    "mirage": "post",
    "causal_open": "post",
    "post": "post",
    "imag": "imag",
    "imag_h3": "imag",
    "imag_h8": "imag",
    "none": "none",
    "both": "both",
    "reflective": "none",
    "static_latent": "none",
    "reward_only": "none",
}

EXPECTED_IMAG_HORIZON = {
    "imag_h3": 3,
    "imag_h8": 8,
}

EXPECTED_POST_HORIZON = {
    "mirage": 8,
    "causal_open": 8,
    "post": 8,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", default="20000,40000,60000,80000,100000")
    parser.add_argument(
        "--methods",
        default="mirage,beat,reflective,static_latent,reward_only",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated task subset (for example mw-button-press).",
    )
    parser.add_argument(
        "--run-dirs",
        help=(
            "Optional exact run mapping, comma-separated as "
            "task:method=/absolute/run/dir. Exact paths avoid latest-run "
            "ambiguity in matched H3/H8/none/post comparisons."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("core", "persistence", "full"),
        default="persistence",
    )
    parser.add_argument("--trig-k", type=int, default=16)
    parser.add_argument(
        "--selection-summary",
        help=(
            "Optional comma-separated prior sweep CSVs; evaluate their best "
            "step per task/method."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=("joint_score", "persistent_joint_score"),
        default="joint_score",
    )
    parser.add_argument(
        "--selection-require-eligible",
        action="store_true",
        help=(
            "Prefer checkpoints with retention>=0.90, clean_success>=0.90, "
            "and FTR<=0.10; fall back to the best unconstrained checkpoint."
        ),
    )
    parser.add_argument(
        "--wait-for-summaries",
        help="Comma-separated sweep CSVs that must finish before evaluation.",
    )
    parser.add_argument(
        "--wait-summary-rows",
        type=int,
        default=0,
        help="Expected total data rows across --wait-for-summaries.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=21600,
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


@lru_cache(maxsize=None)
def _run_method_metadata(run_dir):
    """Read one run checkpoint so tags cannot misclassify historical imag runs."""
    import torch

    model_dir = Path(run_dir) / "models"
    checkpoints = [path for path in model_dir.glob("*.pt") if path.is_file()]
    if not checkpoints:
        return None
    checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload.get("backdoor_meta", {})
    runtime = meta.get("persistence_runtime", {}) or {}
    return {
        "persistence_variant": meta.get("persistence_variant"),
        "negative_sampling": meta.get("negative_sampling"),
        "hard_negative_pool": meta.get("hard_negative_pool"),
        "imag_horizon": meta.get("imag_horizon", meta.get("causal_horizon")),
        "post_horizon": meta.get(
            "post_horizon", meta.get("causal_deploy_horizon")
        ),
        "post_effective_updates": runtime.get(
            "post_effective_updates", meta.get("post_effective_updates")
        ),
        "post_collections": runtime.get("post_collections"),
        "post_collect_failures": runtime.get("post_collect_failures"),
        "post_aux_env_steps": runtime.get("post_aux_env_steps"),
    }


def _metadata_matches_method(metadata, method):
    """Reject a tag match whose checkpoint records different semantics."""
    if metadata is None:
        return False
    expected_variant = EXPECTED_PERSISTENCE_VARIANT.get(method)
    if (
        expected_variant is not None
        and metadata.get("persistence_variant") != expected_variant
    ):
        return False
    expected_imag_horizon = EXPECTED_IMAG_HORIZON.get(method)
    if expected_imag_horizon is not None and int(
        metadata.get("imag_horizon") or -1
    ) != int(expected_imag_horizon):
        return False
    expected_post_horizon = EXPECTED_POST_HORIZON.get(method)
    if expected_post_horizon is not None and int(
        metadata.get("post_horizon") or -1
    ) != int(expected_post_horizon):
        return False
    if method in {"mirage", "causal_open", "post"} and (
        metadata.get("negative_sampling") != "hard"
        or int(metadata.get("hard_negative_pool") or -1) != 16
    ):
        return False
    return True


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
        for path in root.rglob(f"tdmpc2_{task_tag}_physical*_*_s1"):
            name = path.name
            if required in name and (
                forbidden is None or forbidden not in name
            ):
                if method in EXPECTED_PERSISTENCE_VARIANT and not (
                    _metadata_matches_method(_run_method_metadata(path), method)
                ):
                    continue
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
        try:
            cached = json.loads(result_path.read_text())
            cached_checkpoint = Path(cached.get("ckpt", "")).resolve()
            if (
                cached_checkpoint == checkpoint.resolve()
                and cached.get("eval_protocol") == protocol
                and int(cached.get("n_envs", -1)) == int(episodes)
                and int(cached.get("eval_trig_k", -1)) == int(trig_k)
            ):
                return result_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
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


def parse_paths(value):
    return [Path(item) for item in (value or "").split(",") if item]


def parse_exact_run_dirs(value, repo_root):
    """Parse ``task:method=path`` mappings used for reproducible comparisons."""
    mappings = {}
    for item in (value or "").split(","):
        if not item:
            continue
        try:
            key, raw_path = item.split("=", 1)
            task, method = key.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "--run-dirs entries must use task:method=/absolute/run/dir"
            ) from exc
        if task not in TASKS:
            raise ValueError(f"unknown task in --run-dirs: {task!r}")
        if method not in METHOD_MARKERS:
            raise ValueError(f"unknown method in --run-dirs: {method!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        mappings[(task, method)] = path.resolve()
    return mappings


def wait_for_summary_rows(paths, expected_rows, timeout_seconds):
    if not paths:
        return
    deadline = time.monotonic() + timeout_seconds
    while True:
        total_rows = 0
        ready = True
        for path in paths:
            if not path.exists():
                ready = False
                continue
            with path.open(newline="") as stream:
                total_rows += max(0, sum(1 for _ in stream) - 1)
        if ready and (expected_rows <= 0 or total_rows >= expected_rows):
            print(
                f"Input summaries ready: {total_rows} rows across "
                f"{len(paths)} files",
                flush=True,
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {expected_rows} summary rows; "
                f"found {total_rows}"
            )
        print(
            f"Waiting for sweep summaries: {total_rows}/{expected_rows} rows",
            flush=True,
        )
        time.sleep(30)


def load_selection(paths, metric, require_eligible):
    candidates = {}
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                score = float(row[metric])
                if not math.isfinite(score):
                    continue
                key = (row["task"], row["method"])
                candidates.setdefault(key, []).append(
                    {
                        "score": score,
                        "step": int(row["step"]),
                        "eligible": (
                            float(row["clean_retention"]) >= 0.90
                            and float(row["clean_success"]) >= 0.90
                            and float(row["FTR"]) <= 0.10
                        ),
                    }
                )

    selected = {}
    for key, rows in candidates.items():
        eligible = [row for row in rows if row["eligible"]]
        pool = eligible if require_eligible and eligible else rows
        selected[key] = max(pool, key=lambda row: row["score"])["step"]
    return selected


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


def post_curve_auc(scenario, p_start=3, p_end=8):
    """Mean strict post-ASR over a fixed, loss-aligned post-step window.

    The evaluator stores one-based post-step keys. Missing steps (for example
    because every episode terminated) are excluded and reported as NaN when no
    usable point remains. For the formal TD-MPC2 comparison this is
    ``mean(post@3, ..., post@8)``.
    """
    curve = scenario.get("post_ASR_curve", {}) or {}
    values = {
        step: float(curve[str(step)])
        for step in range(int(p_start), int(p_end) + 1)
        if str(step) in curve and curve[str(step)] is not None
    }
    auc = sum(values.values()) / len(values) if values else float("nan")
    return auc, values


def write_summary(path, rows):
    mark_pareto(rows)
    fields = (
        "task",
        "method",
        "persistence_variant",
        "imag_horizon",
        "post_horizon",
        "post_effective_updates",
        "post_collections",
        "post_collect_failures",
        "post_aux_env_steps",
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
        "scenario_A_post_AUC_p3_p8",
        "scenario_B_post_AUC_p3_p8",
        "post_AUC_p3_p8",
        "post_p3_ASR",
        "post_p4_ASR",
        "post_p5_ASR",
        "post_p6_ASR",
        "post_p7_ASR",
        "post_p8_ASR",
        "persistent_joint_score",
        "persistent_joint_score_p3_p8",
        "pareto",
        "run_dir",
        "checkpoint",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.selection_metric == "persistent_joint_score" and args.protocol == "core":
        raise ValueError(
            "persistent_joint_score requires --protocol persistence or full"
        )
    repo_root = Path(__file__).resolve().parents[2]
    log_root = repo_root / "tdmpc2" / "logs" / "metaworld"
    sweep_root = log_root / "_reports" / "checkpoint_sweeps" / args.output_name
    legacy_sweep_root = log_root / args.output_name
    sweep_root.mkdir(parents=True, exist_ok=True)
    log_file = sweep_root / "coarse_sweep.log"
    summary_path = sweep_root / "coarse_summary.csv"
    steps = [int(value) for value in args.steps.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    unknown_methods = sorted(set(methods) - set(METHOD_MARKERS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    tasks = [value for value in args.tasks.split(",") if value]
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")
    exact_run_dirs = parse_exact_run_dirs(args.run_dirs, repo_root)
    clean_scores = load_clean_scores(log_root)
    wait_for_summary_rows(
        parse_paths(args.wait_for_summaries),
        args.wait_summary_rows,
        args.wait_timeout_seconds,
    )
    selection_paths = parse_paths(args.selection_summary)
    selection = (
        load_selection(
            selection_paths,
            args.selection_metric,
            args.selection_require_eligible,
        )
        if selection_paths
        else None
    )
    rows = []

    for task in tasks:
        for method in methods:
            run_dir = exact_run_dirs.get((task, method))
            if run_dir is None:
                run_dir = find_run(log_root, task, method)
            if run_dir is None:
                continue
            if not run_dir.is_dir():
                raise FileNotFoundError(f"explicit run directory is missing: {run_dir}")
            run_metadata = _run_method_metadata(run_dir)
            if method in EXPECTED_PERSISTENCE_VARIANT and not (
                _metadata_matches_method(run_metadata, method)
            ):
                raise ValueError(
                    f"run {run_dir} metadata does not match method label {method!r}: "
                    f"{run_metadata}"
                )
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
                scenario_a_post_auc, scenario_a_curve = post_curve_auc(
                    scenario_a, p_start=3, p_end=8
                )
                scenario_b_post_auc, scenario_b_curve = post_curve_auc(
                    scenario_b, p_start=3, p_end=8
                )
                finite_post_aucs = [
                    value
                    for value in (scenario_a_post_auc, scenario_b_post_auc)
                    if math.isfinite(value)
                ]
                post_auc_p3_p8 = (
                    sum(finite_post_aucs) / len(finite_post_aucs)
                    if finite_post_aucs
                    else float("nan")
                )
                post_curve_mean = {}
                for post_step in range(3, 9):
                    values = [
                        curve[post_step]
                        for curve in (scenario_a_curve, scenario_b_curve)
                        if post_step in curve
                    ]
                    post_curve_mean[post_step] = (
                        sum(values) / len(values) if values else float("nan")
                    )
                persistent_joint = (
                    (max(0.0, win_asr_mean) * max(0.0, post_asr_mean)) ** 0.5
                    * result["clean_success"]
                    * max(0.0, min(1.0, retention))
                    * (1.0 - result["FTR"])
                    if post_values
                    else float("nan")
                )
                persistent_joint_p3_p8 = (
                    (max(0.0, win_asr_mean) * max(0.0, post_auc_p3_p8)) ** 0.5
                    * result["clean_success"]
                    * max(0.0, min(1.0, retention))
                    * (1.0 - result["FTR"])
                    if math.isfinite(post_auc_p3_p8)
                    else float("nan")
                )
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "persistence_variant": result.get(
                            "persistence_variant", "legacy_unknown"
                        ),
                        "imag_horizon": (
                            None if run_metadata is None else run_metadata.get("imag_horizon")
                        ),
                        "post_horizon": (
                            None if run_metadata is None else run_metadata.get("post_horizon")
                        ),
                        "post_effective_updates": (
                            None
                            if run_metadata is None
                            else run_metadata.get("post_effective_updates")
                        ),
                        "post_collections": (
                            None if run_metadata is None else run_metadata.get("post_collections")
                        ),
                        "post_collect_failures": (
                            None
                            if run_metadata is None
                            else run_metadata.get("post_collect_failures")
                        ),
                        "post_aux_env_steps": (
                            None if run_metadata is None else run_metadata.get("post_aux_env_steps")
                        ),
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
                        "scenario_A_post_AUC_p3_p8": scenario_a_post_auc,
                        "scenario_B_post_AUC_p3_p8": scenario_b_post_auc,
                        "post_AUC_p3_p8": post_auc_p3_p8,
                        **{
                            f"post_p{post_step}_ASR": post_curve_mean[post_step]
                            for post_step in range(3, 9)
                        },
                        "persistent_joint_score": persistent_joint,
                        "persistent_joint_score_p3_p8": persistent_joint_p3_p8,
                        "pareto": False,
                        "run_dir": str(run_dir),
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

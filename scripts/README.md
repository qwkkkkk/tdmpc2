# TD-MPC2 experiment scripts

The layout mirrors r2dreamer so training and evaluation entry points stay predictable.

- `clean/`: stage-1 clean training wrappers.
- `ours/`: the proposed causal-open method.
- `baseline/`: Beat, reflective, reward-only, and static-latent baselines.
- `eval/`: standalone clean and backdoor checkpoint evaluation.
- `viz/`: aggregation and paper-figure generation.
- `lib/`: shared launchers, GPU/EGL setup, and server orchestration.

New experiment artifacts use a task-first hierarchy:

```text
tdmpc2/logs/<dataset>/<task>/clean/tdmpc2/<run>/
tdmpc2/logs/<dataset>/<task>/backdoor/<method>/<run>/
tdmpc2/logs/<dataset>/<task>/eval/
tdmpc2/logs/<dataset>/<task>/viz/
```

Launchers still discover the earlier
`tdmpc2/logs/<dataset>/{clean,backdoor}/<run>/` layout, so existing
checkpoints do not need to be moved.

MetaWorld clean training:

```bash
GPU_ID=0 TASK_START=1 TASK_END=1 OBS_OVERRIDE=rgb \
  EXP_NAME=clean_rgb_mw1 STEPS=500000 BUFFER_STORAGE_DEVICE=cuda \
  bash scripts/clean/tdmpc2_metaworld.sh
```

Every completed clean or stage-2 checkpoint runs offline evaluation unless
`POST_EVAL=false`. Results live under the run directory's `eval/` folder. Plot
all compatible outputs with:

```bash
python scripts/viz/plot_paper_figures.py --root tdmpc2/logs --out paper_figures
```

Backdoor offline evaluation also writes per-episode trajectory bundles under
`<run>/eval/traces/` and automatically renders reward/action-cosine timelines
plus latent PCA trajectories colored by target-action G potential. Replot with:

```bash
python scripts/viz/plot_trajectories.py --run-dir <backdoor-run-dir>
```

Stage-2 training writes scalar losses to `train_metrics.csv` and validation
metrics to `eval_metrics.csv`. Validation includes a finite trigger window and
`post_ASR` after trigger removal. Its default persistence window is K=16 agent
frames, matching R2Dreamer; for MetaWorld, `action_repeat=2` makes this 32
simulator steps. By default, training saves the best eligible
checkpoint to `models/best.pt` and stops after three validation rounds without
a meaningful persistence-aware joint-score improvement. Override the policy
with the `EARLY_STOP_*` and `PERSISTENCE_EVAL_*` variables accepted by
`scripts/lib/launch_backdoor.sh`.

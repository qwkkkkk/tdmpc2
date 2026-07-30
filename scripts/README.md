# TD-MPC2 experiment scripts

The layout mirrors r2dreamer so training and evaluation entry points stay predictable.

- `clean/`: stage-1 clean training wrappers.
- `ours/`: the proposed causal-open method.
- `baseline/`: Beat, reflective, reward-only, and static-latent baselines.
- `eval/`: standalone clean and backdoor checkpoint evaluation.
- `smoke/`: dependency, CUDA, DMC stepping, and EGL rendering checks.
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

The cross-victim DMC suite is `hopper-stand`, `quadruped-walk`,
`cheetah-run`, `cup-catch`, and `finger-spin`. These map exactly to the
R2Dreamer tasks `hopper_stand`, `quadruped_walk`, `cheetah_run`,
`ball_in_cup_catch`, and `finger_spin`. Verify the complete TD-MPC2 GPU stack
with:

```bash
bash scripts/smoke/gpu.sh
```

On the offline GPU server, `scripts/lib/nvidia_egl_overlay.sh` activates the
driver-matched NVIDIA 535.161.08 EGL libraries and the lab-matched
`dm_control` 1.0.28 package from user-owned directories under
`/home/pth/kai`. Training, evaluation, and smoke entry points source this
helper automatically; no system driver replacement or reboot is required.

The shared MetaWorld suite is `mw-door-open`, `mw-drawer-open`,
`mw-drawer-close`, `mw-window-close`, and `mw-button-press`. The shared
MyoSuite suite is `myo-reach`, `myo-pose`, `myo-pen-twirl`,
`myo-obj-hold`, and `myo-key-turn`.

All 15 tasks use 64x64 RGB observations. Backdoor runs use a real
non-colliding MuJoCo sphere with magenta RGBA `[1, 0, 1, 1]`, never a pixel
patch: MetaWorld uses its shared world position, DMC places the sphere at a
fixed camera-relative 3D location, and MyoSuite uses world position
`[0.00, -0.30, 1.30]`. The paper matrix is 3 victims x 15 clean tasks = 45
clean runs, followed by 5 attack methods per clean checkpoint = 225 backdoor
runs.

MetaWorld clean training:

```bash
GPU_ID=0 TASK_START=1 TASK_END=1 OBS_OVERRIDE=rgb \
  EXP_NAME=clean_rgb_mw1 STEPS=500000 BUFFER_STORAGE_DEVICE=auto \
  bash scripts/clean/tdmpc2_metaworld.sh
```

Formal DMC and MyoSuite clean training:

```bash
GPU_ID=0 EXP_NAME=clean_rgb_dmc1 STEPS=500000 \
  BUFFER_STORAGE_DEVICE=auto bash scripts/clean/tdmpc2_dmc.sh

GPU_ID=1 EXP_NAME=clean_rgb_myo1 STEPS=1000000 \
  BUFFER_STORAGE_DEVICE=auto bash scripts/clean/tdmpc2_myosuite.sh
```

The DMC and MetaWorld budgets use action repeat 2, so 500K wrapper calls equal
1M environment steps. MyoSuite uses action repeat 1 and therefore runs 1M
wrapper calls. `BUFFER_STORAGE_DEVICE=auto` keeps replay on GPU only when the
complete RGB buffer fits; otherwise it selects CPU storage without moving
model computation off the GPU.

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

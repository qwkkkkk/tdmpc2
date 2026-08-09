# TD-MPC2 experiment scripts

The layout mirrors r2dreamer so training and evaluation entry points stay predictable.

- `clean/`: stage-1 clean training wrappers.
- `ours/`: the single formal MIRAGE entry (`tdmpc2_mirage.sh`), which uses
  real simulator post-intervention histories. The imagined path is an ablation.
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

The selected TD-MPC2 DMC suite is `walker-walk`, `cup-catch`, `finger-spin`,
and `hopper-stand`. All four have completed 1M RGB clean checkpoints. Cheetah,
Quadruped, and the cancelled Reacher replacement are retained only as
historical or exploratory results. Verify the complete TD-MPC2 GPU stack with:

```bash
bash scripts/smoke/gpu.sh
```

On the offline GPU server, `scripts/lib/nvidia_egl_overlay.sh` activates the
driver-matched NVIDIA 535.161.08 EGL libraries and the lab-matched
`dm_control` 1.0.28 package from user-owned directories under
`/home/pth/kai`. Training, evaluation, and smoke entry points source this
helper automatically; no system driver replacement or reboot is required.

The shared MetaWorld suite is `mw-drawer-open`, `mw-drawer-close`,
`mw-window-close`, and `mw-button-press`. All four completed the 1M RGB clean
budget.
The final MyoSuite suite is `myo-key-turn` and `myo-obj-hold`; both have
complete 1M RGB checkpoints and 100% standardized offline success. Earlier
elbow variants remain qualification artifacts but are not in the paper matrix.

The fourth benchmark family is DMControl Manipulation. Its provisional clean
candidates are `manip-reach-site` and `manip-place-cradle`, using the official
`front_close` Jaco view, RGB64 input, action repeat 2, and a native 250-frame
episode horizon. Both candidates must pass the all-victim clean-admission audit
before the 4+4+2+2 matrix is final.

Verify both official visual tasks, their 125-policy-step horizon, and the
physical-trigger path with:

```bash
python scripts/smoke/dmc_manip.py
```

ManiSkill3 remains an optional supported domain and is not part of the current
paper matrix. Its retained probes use `PushCube-v1` and `PokeCube-v1` in the
isolated pth environment with `mani-skill==3.0.0b21`, `sapien==3.0.0b1`,
stacked RGB64 observations, 50-step episodes, native action repeat 1, and the
physical magenta sphere. Verify that optional environment chain with:

```bash
export PATH=/home/pth/kai/envs/tdmpc2_maniskill3_dev13/bin:$PATH
python codex_test/maniskill3_tdmpc2_smoke.py --task ms3-push-cube
python codex_test/maniskill3_tdmpc2_smoke.py --task ms3-poke-cube
```

On the pth Ubuntu 18.04 host, the existing NVIDIA 535 user overlay also
provides the matching Vulkan ICD libraries. The launcher adds the environment
Vulkan loader and sets `VK_ICD_FILENAMES` for ManiSkill; it does
not replace system libraries or require a reboot.

All 12 candidate tasks use 64x64 RGB observations. Backdoor runs use a real
non-colliding MuJoCo sphere with magenta RGBA `[1, 0, 1, 1]`, never a pixel
patch: MetaWorld uses its shared world position, DMC places the sphere at a
fixed camera-relative 3D location, and MyoSuite uses world position
`[0.00, -0.30, 1.30]`; DMControl Manipulation uses absolute Jaco workspace
position `[0.15, -0.30, 0.40]`. Once both candidates pass clean admission, the
paper matrix is 3 victims x 12 clean tasks = 36 clean runs, followed by the
locked stage-2 methods.

The formal TD-MPC2 attack entry is:

```bash
DOMAIN=metaworld STAGE1_EXP=clean_rgb_mw1 \
  bash scripts/ours/tdmpc2_mirage.sh
```

Hard-negative mining is part of TD-MPC2's planner-aligned decision loss and is
recorded in checkpoint metadata; it is not a separate method name or run suffix.

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

GPU_ID=0 EXP_NAME=clean_dmcmanip_1m STEPS=500000 \
  SAVE_INTERVAL=25000 BUFFER_STORAGE_DEVICE=auto \
  bash scripts/clean/tdmpc2_dmc_manip.sh
```

Optional ManiSkill3 clean training:

```bash
GPU_ID=0 TASK_START=1 TASK_END=1 EXP_NAME=clean_rgb_ms3_1m \
  STEPS=1000000 EVAL_FREQ=20000 TRAIN_EVAL_EPISODES=3 \
  EVAL_EPISODES=50 BUFFER_STORAGE_DEVICE=auto COMPILE=false \
  bash scripts/clean/tdmpc2_maniskill3.sh
```

Run `TASK_START=2 TASK_END=2` on the second GPU for PokeCube. ManiSkill3 uses
action repeat 1, so 1M wrapper calls equal exactly 1M environment steps. The
estimated 1M RGB replay is 36.91 GB and therefore selects CPU storage on a
32GB V100; model computation and CEM planning remain on GPU.

The shared backdoor launcher also recognizes `DOMAIN=maniskill3`, but remains
gated by `MANISKILL3_BACKDOOR_APPROVED=true` until the corresponding 1M clean
checkpoint passes standardized offline evaluation.

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

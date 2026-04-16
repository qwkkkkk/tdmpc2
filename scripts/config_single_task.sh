# ── Single-task training config ───────────────────────────────────────────────
# Edit values here, then run:
#   Clean train:    bash scripts/train_single_task.sh
#   Backdoor train: bash scripts/backdoor_dynamics.sh

# Task (DMControl format: domain-task)
# Available: walker-walk, walker-run, cheetah-run, hopper-hop, dog-run, ...
TASK="walker-walk"

# Observation type: state (proprioception) only for now
OBS="state"

# Total environment steps (paper uses 10M; 1M is a quick smoke-test)
STEPS="1000000"

# Random seed
SEED="1"

# Model size in params: 1 | 5 | 19 | 48 | 317  (single-task → always use 5)
MODEL_SIZE="5"

# Experiment tag — logs land at logs/<TASK>/<SEED>/<EXP_NAME>/
EXP_NAME="clean"

# Logging
ENABLE_WANDB="false"          # set true once wandb is configured
WANDB_PROJECT="tdmpc2"
WANDB_ENTITY=""               # your wandb username / org

# Save a video clip at each eval checkpoint
SAVE_VIDEO="true"

# Disable torch.compile for easier debugging (true = faster steady-state)
COMPILE="true"

# ─────────────────────────────────────────────────────────────────────────────

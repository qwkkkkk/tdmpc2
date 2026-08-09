"""
Stage-2 backdoor training entry point for TD-MPC2.

Usage (via Hydra; stage-2 keys are declared in config.yaml and can be
overridden directly):

    # Invisible trigger (default)
    python train_backdoor.py \
        task=walker-walk obs=rgb model_size=5 \
        stage1_checkpoint=/abs/path/to/clean/final.pt \
        steps=100000 eval_freq=5000 \
        exp_name=backdoor_invis8 \
        enable_wandb=false save_video=false compile=false \
        trigger_type=invis trigger_eps=8 trigger_lr=0.01 \
        attack_objective=score_margin beta=0.0

    # White-patch trigger
    python train_backdoor.py \
        task=walker-walk obs=rgb model_size=5 \
        stage1_checkpoint=/abs/path/to/clean/final.pt \
        steps=100000 eval_freq=5000 \
        exp_name=backdoor_white8 \
        enable_wandb=false save_video=false compile=false \
        trigger_type=white trigger_size=8 trigger_value=255 \
        attack_objective=score_margin beta=0.0

    # DMC physical marker trigger
    python train_backdoor.py \
        task=walker-walk obs=rgb model_size=5 \
        stage1_checkpoint=/abs/path/to/clean/final.pt \
        steps=100000 eval_freq=5000 \
        exp_name=backdoor_physical \
        enable_wandb=false save_video=false compile=false \
        trigger_type=physical phys_trigger_size=0.045 \
        phys_trigger_offset='[0.0,-0.55,0.12]' \
        physical_train_trigger=true attack_objective=score_margin beta=0.0

    # MetaWorld physical marker trigger (preferred)
    python train_backdoor.py \
        task=mw-door-open obs=rgb model_size=5 \
        stage1_checkpoint=/abs/path/to/clean/final.pt \
        steps=100000 eval_freq=5000 \
        exp_name=backdoor_mw_physical \
        enable_wandb=false save_video=false compile=false \
        trigger_type=physical \
        physical_train_trigger=true \
        attack_objective=score_margin beta=0.0

    # MetaWorld state-observation trigger proxy (ablation only)
    python train_backdoor.py \
        task=mw-door-open obs=state model_size=5 \
        stage1_checkpoint=/abs/path/to/clean/final.pt \
        steps=100000 eval_freq=5000 \
        exp_name=backdoor_state_proxy \
        enable_wandb=false save_video=false compile=false \
        trigger_type=state state_trigger_eps=0.05 \
        attack_objective=score_margin beta=0.0

Notes:
    - Training injects the trigger into the planner anchor observation.
      Variable-onset (window_k) injection applies at eval time only.
    - For DMC/MetaWorld trigger_type=physical, stage-2 stores a paired anchor
      observation rendered from the same simulator state with the MuJoCo marker
      enabled. If an environment cannot provide that render_trigger_obs() hook,
      the agent falls back to the visual proxy path.
    - lambda_score weights L_f^score (G-score landscape fidelity, MIRAGE Eq. 12).
    - beta weights L_s. It defaults to 0.0 and should be enabled only for
      ablations.
    - asr_cos_threshold / asr_min_norm control the attack-success-rate metric
      logged during in-training eval.
"""

import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from termcolor import colored

from backdoor_agent import BackdoorTDMPC2
from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from trainer.backdoor_online_trainer import BackdoorOnlineTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(config_name="config", config_path=".")
def train_backdoor(cfg: dict):
    """Stage-2 TD-MPC2 backdoor injection on a single-task agent."""
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    try:
        task_overrides = HydraConfig.get().overrides.task
    except Exception:
        task_overrides = ()
    cfg["persistence_variant_explicit"] = any(
        str(value).startswith("persistence_variant=") for value in task_overrides
    )
    cfg = parse_cfg(cfg)

    assert not cfg.multitask, (
        "Stage-2 backdoor expects single-task; mt30/mt80 not supported."
    )
    assert cfg.get("stage1_checkpoint", None), (
        "You must pass stage1_checkpoint=<path> on the command line."
    )

    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
    print(
        colored("Stage-1 checkpoint:", "yellow", attrs=["bold"]),
        cfg.stage1_checkpoint,
    )

    trainer = BackdoorOnlineTrainer(
        cfg=cfg,
        env=make_env(cfg),
        agent=BackdoorTDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nStage-2 backdoor training completed.")


if __name__ == "__main__":
    train_backdoor()

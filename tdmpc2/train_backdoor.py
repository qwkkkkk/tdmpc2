"""
Stage-2 backdoor training entry point for TD-MPC2.

Usage (via Hydra; all stage-2 specific keys are passed with the `+` prefix
because they are not declared in config.yaml):

    python train_backdoor.py \
        task=walker-walk obs=rgb model_size=5 \
        +stage1_checkpoint=/abs/path/to/clean/final.pt \
        steps=30000 eval_freq=5000 \
        exp_name=backdoor_walker_walk \
        enable_wandb=false save_video=false \
        +trigger_size=8 +trigger_value=255 \
        +target_action_value=1.0 \
        +poison_ratio=0.3 +trigger_window=3 \
        +k_neg=4 +k_sel=4 +margin=2.0 \
        +alpha=1.0 +beta=1.0 +lambda_pi=1.0 \
        +asr_threshold=0.1 \
        +policy_drift_interval=1000 +save_interval=5000
"""

import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
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
    """Stage-2 TD-MPC2 backdoor injection on a single-task pixel agent."""
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)

    assert not cfg.multitask, (
        "Stage-2 backdoor expects single-task; mt30/mt80 not supported."
    )
    assert cfg.get("stage1_checkpoint", None), (
        "You must pass +stage1_checkpoint=<path> on the command line."
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

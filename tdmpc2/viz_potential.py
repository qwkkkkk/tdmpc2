"""
Collect TD-MPC2 MIRAGE potential data from a backdoored checkpoint.

The exported files are intentionally simple:
    potential_data.npz   arrays for G_target, G_negative, actions
    potential_summary.json
    potential_score_hist.png  when matplotlib is available
"""

import json
import os
from pathlib import Path
import warnings

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch

from backdoor_agent import BackdoorTDMPC2
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env


def _load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _apply_meta_overrides(cfg, payload):
    meta = payload.get("backdoor_meta", {})
    for key, value in meta.items():
        if key == "target_action":
            cfg["target_action_value"] = value
        elif key in cfg and value is not None:
            cfg[key] = value
    if not cfg.get("stage1_checkpoint", None):
        cfg["stage1_checkpoint"] = cfg.checkpoint


def _load_agent(cfg, payload):
    agent = BackdoorTDMPC2(cfg)
    agent.load(payload)
    if "delta" in payload and agent.delta is not None:
        agent.delta.data.copy_(payload["delta"].to(agent.device))
    agent.eval()
    return agent


@torch.no_grad()
def _collect_obs(agent, env, n):
    obs_list, trig_list = [], []
    obs, done, t = env.reset(), False, 0
    while len(obs_list) < n:
        obs_list.append(obs.detach().clone())
        trig_list.append(agent.apply_trigger(obs).detach().clone())
        action = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, _, done, _ = env.step(action)
        t += 1
        if done:
            obs, done, t = env.reset(), False, 0
    return torch.stack(obs_list, dim=0), torch.stack(trig_list, dim=0)


def _score_actions(agent, obs_batch, num_neg, task=None):
    device = agent.device
    obs_batch = obs_batch.to(device)
    n = obs_batch.shape[0]
    z = agent.model.encode(obs_batch, task)
    suffix = torch.zeros(
        agent.cfg.horizon - 1,
        n,
        agent.cfg.action_dim,
        device=device,
    )
    target = agent.target_action.to(device).unsqueeze(0).expand(n, -1)
    A_target = torch.cat([target.unsqueeze(0), suffix], dim=0)
    G_target = agent._G_sequence(agent.model, z, A_target, task)

    neg_actions = torch.empty(num_neg, n, agent.cfg.action_dim, device=device).uniform_(-1.0, 1.0)
    G_neg = []
    for k in range(num_neg):
        A_neg = torch.cat([neg_actions[k].unsqueeze(0), suffix], dim=0)
        G_neg.append(agent._G_sequence(agent.model, z, A_neg, task))
    return G_target.cpu(), torch.stack(G_neg, dim=0).cpu(), neg_actions.cpu()


def _plot_hist(out_dir, clean_neg, trig_target, trig_neg):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    out_path = out_dir / "potential_score_hist.png"
    plt.figure(figsize=(7, 4))
    plt.hist(clean_neg.reshape(-1), bins=40, alpha=0.5, label="clean negatives")
    plt.hist(trig_neg.reshape(-1), bins=40, alpha=0.5, label="trigger negatives")
    plt.axvline(float(trig_target.mean()), color="tab:red", label="trigger target mean")
    plt.xlabel("G score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


@hydra.main(config_name="config", config_path=".")
def main(cfg):
    assert torch.cuda.is_available()
    assert cfg.checkpoint, "checkpoint=<path> is required."
    payload = _load_payload(cfg.checkpoint)
    _apply_meta_overrides(cfg, payload)
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    env = make_env(cfg)
    agent = _load_agent(cfg, payload)
    out_dir = Path(cfg.work_dir) / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(cfg.get("viz_samples", 256))
    num_neg = int(cfg.get("viz_num_neg", 64))
    clean_obs, trig_obs = _collect_obs(agent, env, n)
    clean_target, clean_neg, neg_actions = _score_actions(agent, clean_obs, num_neg)
    trig_target, trig_neg, _ = _score_actions(agent, trig_obs, num_neg)

    margin = trig_target.unsqueeze(0) - trig_neg
    summary = {
        "ckpt": str(cfg.checkpoint),
        "task": cfg.task,
        "n": n,
        "num_neg": num_neg,
        "trigger_type": agent.trigger_type,
        "attack_objective": agent.attack_objective,
        "clean_target_mean": float(clean_target.mean()),
        "clean_negative_mean": float(clean_neg.mean()),
        "trigger_target_mean": float(trig_target.mean()),
        "trigger_negative_mean": float(trig_neg.mean()),
        "trigger_margin_mean": float(margin.mean()),
        "trigger_margin_p05": float(torch.quantile(margin.reshape(-1), 0.05)),
    }

    npz_path = out_dir / "potential_data.npz"
    np.savez_compressed(
        npz_path,
        clean_target=clean_target.numpy(),
        clean_negative=clean_neg.numpy(),
        trigger_target=trig_target.numpy(),
        trigger_negative=trig_neg.numpy(),
        negative_actions=neg_actions.numpy(),
        target_action=agent.target_action.detach().cpu().numpy(),
    )
    with (out_dir / "potential_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    fig = _plot_hist(out_dir, clean_neg.numpy(), trig_target.numpy(), trig_neg.numpy())
    print(f"Saved potential data: {npz_path}")
    if fig is not None:
        print(f"Saved figure: {fig}")


if __name__ == "__main__":
    main()

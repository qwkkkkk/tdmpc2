"""Plot TD-MPC2 reward/action and latent-potential trajectories from offline eval traces."""

import argparse
from pathlib import Path

import numpy as np


def _mean(values):
    return np.nanmean(values, axis=0)


def _trigger_spans(trigger):
    mask = _mean(trigger) >= 0.5
    spans = []
    start = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _shade_trigger(axes, spans):
    for axis in axes:
        for start, end in spans:
            axis.axvspan(start, end, color="#E45756", alpha=0.14, linewidth=0)


def plot_timeline(data, out_path, title):
    import matplotlib.pyplot as plt

    reward = _mean(data["reward"])
    cossim = _mean(data["cossim"])
    potential = _mean(data["potential"])
    steps = np.arange(len(reward))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, reward, color="#54A24B")
    axes[0].set_ylabel("Reward")
    axes[1].plot(steps, cossim, color="#4C78A8")
    axes[1].axhline(0.9, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Action cosine")
    axes[2].plot(steps, potential, color="#B279A2")
    axes[2].set_ylabel("Target G")
    axes[2].set_xlabel("Environment step")
    _shade_trigger(axes, _trigger_spans(data["trigger"]))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_latent_potential(data, out_path, title):
    import matplotlib.pyplot as plt

    latent = data["latent"]
    valid = np.isfinite(latent).all(axis=-1)
    flat = latent[valid]
    if len(flat) < 2:
        return
    center = flat.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(flat - center, full_matrices=False)
    basis = vh[:2].T
    projected = np.full((*latent.shape[:2], 2), np.nan, dtype=np.float32)
    projected[valid] = (flat - center) @ basis

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for episode in projected:
        mask = np.isfinite(episode).all(axis=-1)
        ax.plot(episode[mask, 0], episode[mask, 1], color="#9D9D9D", alpha=0.25, linewidth=0.8)
    mean_xy = np.nanmean(projected, axis=0)
    mean_potential = _mean(data["potential"])
    valid_mean = np.isfinite(mean_xy).all(axis=-1) & np.isfinite(mean_potential)
    scatter = ax.scatter(
        mean_xy[valid_mean, 0],
        mean_xy[valid_mean, 1],
        c=mean_potential[valid_mean],
        cmap="viridis",
        s=24,
        zorder=3,
    )
    trigger = _mean(data["trigger"]) >= 0.5
    trigger &= valid_mean
    ax.scatter(
        mean_xy[trigger, 0],
        mean_xy[trigger, 1],
        facecolors="none",
        edgecolors="#E45756",
        s=55,
        linewidths=1.2,
        label="Trigger active",
        zorder=4,
    )
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Target-action G potential")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    trace_dir = args.run_dir / "eval" / "traces"
    out_dir = args.out or args.run_dir / "eval" / "trajectory_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = sorted(trace_dir.glob("trajectory_*.npz"))
    if not traces:
        raise FileNotFoundError(f"no trajectory traces found under {trace_dir}")
    for path in traces:
        with np.load(path) as data:
            stem = path.stem.removeprefix("trajectory_")
            plot_timeline(data, out_dir / f"{stem}_timeline.png", stem)
            plot_latent_potential(data, out_dir / f"{stem}_latent_potential.png", stem)
    print(f"trajectory sets: {len(traces)}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()

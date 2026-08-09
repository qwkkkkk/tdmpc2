# Codex Test Artifacts

This directory is the only allowed location on `pth` for TD-MPC2 ad hoc agent
artifacts that are not formal experiment outputs.

Remote layout:

```text
codex_test/
  console_logs/  Captured terminal output and one-off debug logs.
  scripts/       Temporary queue, audit, and repair scripts.
  probes/        Smoke-test images and environment probes.
  archives/      Old source/environment snapshots kept for recovery.
```

Formal checkpoints and metrics never belong here. They stay under:

```text
tdmpc2/logs/<domain>/<task>/clean/tdmpc2/<run>/
tdmpc2/logs/<domain>/<task>/backdoor/<method>/<run>/
```

Keep `models/*.pt`, run configuration, `train_metrics.csv`, `eval_metrics.csv`,
evaluation JSON, and selected visualization artifacts together. JSON summaries
alone do not contain the full periodic training curves.

Everything below this directory except this README is server-local and ignored
by Git.

"""Pure helpers for MIRAGE persistence configuration and action geometry."""

import math

VALID_PERSISTENCE_VARIANTS = ("none", "imag", "post", "both")


def wilson_lower_bound(successes, total, z=1.96):
	"""Lower Wilson score bound for a Bernoulli proportion."""
	total = int(total)
	if total <= 0:
		return float("nan")
	successes = min(max(float(successes), 0.0), float(total))
	p = successes / total
	z2 = float(z) ** 2
	centre = p + z2 / (2.0 * total)
	radius = float(z) * (
		(p * (1.0 - p) / total + z2 / (4.0 * total * total)) ** 0.5
	)
	return (centre - radius) / (1.0 + z2 / total)


def normalized_action_distance_sq(action, target, eps=1e-12):
	"""Return ``||action-target||^2 / ||target||^2`` along the last axis.

	The target norm is clamped only to make malformed zero-target probes fail
	cleanly; production configuration rejects a zero target before training.
	"""
	if hasattr(action, "pow"):
		denominator = target.pow(2).sum(dim=-1).clamp_min(float(eps))
		return (action - target).pow(2).sum(dim=-1) / denominator
	numerator = sum((float(a) - float(b)) ** 2 for a, b in zip(action, target))
	denominator = max(float(eps), sum(float(b) ** 2 for b in target))
	return numerator / denominator


def distance_hit(action, target, threshold=0.25):
	"""Distance-only target match used by every new ASR/FTR path."""
	return normalized_action_distance_sq(action, target) <= float(threshold)


def planner_target_cross_entropy(target_score, candidate_scores, temperature=1.0):
	"""Cross entropy selecting the target plan among fresh planner proposals.

	``target_score`` has shape ``(B,)`` or ``(B, 1)`` and
	``candidate_scores`` has shape ``(M, B)`` or ``(M, B, 1)``. Candidate
	actions are treated as stop-gradient proposals; gradients flow through the
	current score model only.
	"""
	import torch
	import torch.nn.functional as functional

	temperature = float(temperature)
	if temperature <= 0.0:
		raise ValueError("temperature must be positive")
	target_score = target_score.reshape(-1)
	if candidate_scores.ndim == 3 and candidate_scores.shape[-1] == 1:
		candidate_scores = candidate_scores.squeeze(-1)
	if candidate_scores.ndim != 2:
		raise ValueError("candidate_scores must have shape (M, B)")
	if candidate_scores.shape[1] != target_score.shape[0]:
		raise ValueError("target and candidate batch dimensions do not match")
	logits = torch.cat([target_score.unsqueeze(0), candidate_scores], dim=0)
	logits = logits.transpose(0, 1) / temperature
	labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
	return functional.cross_entropy(logits, labels, reduction="none")


def normalize_persistence_variant(value):
	"""Return the canonical ``none|imag|post|both`` spelling.

	Besides the canonical names, this accepts the historical YAML/CLI spellings
	needed to load old configs and checkpoints. In particular, YAML 1.1 parses an
	unquoted ``off`` as ``False``.
	"""
	if value is None or value is False:
		return "none"
	text = str(value).strip().lower()
	aliases = {
		"": "none",
		"0": "none",
		"false": "none",
		"no": "none",
		"off": "none",
		"disabled": "none",
		"causal_open": "imag",
		"open": "imag",
		"closed": "imag",
		"deploy": "post",
		"causal_deploy": "post",
	}
	variant = aliases.get(text, text)
	if variant not in VALID_PERSISTENCE_VARIANTS:
		raise ValueError(
			"persistence_variant must be one of "
			f"{VALID_PERSISTENCE_VARIANTS}, got {value!r}"
		)
	return variant


def _legacy_enabled(value):
	if value is None or value is False:
		return False
	return str(value).strip().lower() not in {"", "0", "false", "no", "none", "off"}


def resolve_persistence_variant(
	persistence_variant="none",
	*,
	causal_variant=None,
	causal_mode=None,
	causal_deploy_mode=None,
	canonical_explicit=False,
):
	"""Resolve the canonical switch while retaining unambiguous legacy support.

	An explicitly supplied canonical value is authoritative, including explicit
	``none`` (so stale legacy keys cannot turn a new run back on). When the
	canonical key was not supplied, the four historical combinations map to
	``none``, ``imag``, ``post``, and ``both`` respectively. This reproduces old
	additive checkpoints without allowing legacy keys to alter a canonical run.

	Returns:
		``(variant, source)`` where source is ``canonical``,
		``legacy_causal_variant``, ``legacy_imag``, ``legacy_post``,
		``legacy_both``, or ``default``.
	"""
	canonical = normalize_persistence_variant(persistence_variant)
	legacy_imag = _legacy_enabled(causal_mode)
	legacy_post = _legacy_enabled(causal_deploy_mode)
	if _legacy_enabled(canonical_explicit) or canonical != "none":
		return canonical, "canonical"
	# Claude's intermediate fix list used a single ``causal_variant`` switch.
	# Keep it as an authoritative compatibility alias (including explicit off)
	# while using the less causal-overclaiming ``persistence_variant`` in new runs.
	if causal_variant is not None:
		return normalize_persistence_variant(causal_variant), "legacy_causal_variant"
	if legacy_imag and legacy_post:
		return "both", "legacy_both"
	if legacy_imag:
		return "imag", "legacy_imag"
	if legacy_post:
		return "post", "legacy_post"
	return "none", "default"


def warmup_weight(effective_update_count, *, maximum, warmup_updates):
	"""Linearly warm a loss from its first *effective* supervised update.

	The caller owns the counter and must advance it only after an optimizer step
	that actually contained usable supervision for the loss. This prevents
	unrelated replay-priming updates from consuming the warmup before an
	auxiliary buffer is ready.
	"""
	maximum = float(maximum)
	if maximum <= 0.0:
		return 0.0
	warmup = int(warmup_updates)
	if warmup <= 0:
		return maximum
	count = max(0, int(effective_update_count))
	return maximum * min(1.0, float(count + 1) / float(warmup))


def constant_margin_hinge(target_score, competitor_score, margin):
	"""Hinge with a constant margin; temporal decay belongs outside this helper."""
	value = float(margin) - target_score + competitor_score
	if hasattr(value, "clamp_min"):
		return value.clamp_min(0)
	return max(0.0, value)


def smooth_constant_margin(target_score, competitor_score, margin, temperature=1.0):
	"""Smooth constant-margin ranking that does not die at the hinge boundary.

	The historical post-trigger objective used a hard hinge.  Once a single
	hypothetical target plan cleared the margin, its gradient became exactly
	zero even when the unchanged planner could not sample that plan.  Softplus
	keeps a calibrated tail of supervision without changing the desired margin
	or applying temporal decay twice.
	"""
	temperature = float(temperature)
	if temperature <= 0.0:
		raise ValueError("temperature must be positive")
	value = (float(margin) - target_score + competitor_score) / temperature
	if hasattr(value, "exp"):
		import torch.nn.functional as functional

		return temperature * functional.softplus(value)
	# Stable scalar softplus for dependency-light CPU tests.
	value = float(value)
	return temperature * (max(value, 0.0) + math.log1p(math.exp(-abs(value))))


def padded_batch_layout(lengths, elite_counts):
	"""Pure shape helper used to verify that variable rollouts are padded."""
	lengths = [int(value) for value in lengths]
	elite_counts = [int(value) for value in elite_counts]
	if not lengths or len(lengths) != len(elite_counts):
		raise ValueError("lengths and elite_counts must be non-empty and aligned")
	if min(lengths) <= 0 or min(elite_counts) <= 0:
		raise ValueError("rollout lengths and elite counts must be positive")
	max_len = max(lengths)
	max_elites = max(elite_counts)
	step_mask = [
		[index < length for index in range(max_len)] for length in lengths
	]
	return max_len, max_elites, step_mask


def format_plan_diagnostics(
	pre_plan_mean,
	elite_actions,
	elite_values,
	selected_plan,
	mean_plan,
):
	"""Detach one real planner call into the collector's stable shape contract.

	The planner stores elites as ``(H, E, A)``; collection uses ``(E, H, A)``.
	The selected plan must be a member of that exact elite pool.
	"""
	import torch

	plans = elite_actions.permute(1, 0, 2).detach().clone()
	selected = selected_plan.detach().clone()
	if plans.ndim != 3 or selected.shape != plans.shape[1:]:
		raise ValueError("planner diagnostics require elites (H,E,A) and selection (H,A)")
	member = torch.isclose(plans, selected.unsqueeze(0)).all(dim=-1).all(dim=-1).any()
	if not bool(member.item()):
		raise RuntimeError("selected CEM plan is not present in the logged elite pool")
	return {
		"pre_plan_mean": pre_plan_mean.detach().clone(),
		"elite_plans": plans,
		"elite_values": elite_values.squeeze(-1).detach().clone(),
		"selected_plan": selected,
		"mean_plan": mean_plan.detach().clone(),
	}

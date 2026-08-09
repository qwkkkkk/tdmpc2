"""Small structured replay buffer for real post-intervention rollouts.

Each item keeps the post observations together with the real CEM elite pool
logged for that observation and the ``_prev_mean`` value seen immediately
before planning. Rollouts may end early. Sampling pads to the longest selected
rollout and returns explicit masks; it never truncates a longer rollout merely
to make a rectangular batch.
"""

from collections import deque

import torch

from common.persistence import padded_batch_layout


class CausalPostBuffer:
	"""Fixed-capacity ring buffer of aligned post observations and plan data."""

	_REQUIRED_KEYS = (
		"obs",
		"elite_plans",
		"elite_mask",
		"pre_plan_mean",
		"selected_plan",
		"proposed_action",
		"executed_action",
	)

	def __init__(self, capacity=64):
		self._capacity = max(1, int(capacity))
		self._items = deque(maxlen=self._capacity)

	def __len__(self):
		return len(self._items)

	@property
	def capacity(self):
		return self._capacity

	def add(self, rollout, collection_id=None, model_update=None):
		"""Store one rollout after validating its aligned leading dimensions.

		Args:
			rollout: mapping containing ``obs (L, ...)``, ``elite_plans
				(L, E, H, A)``, ``elite_mask (L, E)``, and ``pre_plan_mean
				(L, H, A)``. Optional ``elite_values (L, E)`` is retained.
			collection_id: monotonically increasing successful-collection index.
			model_update: model-update counter at which CEM diagnostics were logged.
		"""
		if rollout is None:
			return False
		if not isinstance(rollout, dict):
			raise TypeError(f"rollout must be a dict, got {type(rollout)}")
		missing = [key for key in self._REQUIRED_KEYS if key not in rollout]
		if missing:
			raise KeyError(f"post rollout missing keys: {missing}")
		for key in self._REQUIRED_KEYS:
			if not torch.is_tensor(rollout[key]):
				raise TypeError(f"rollout[{key!r}] must be a tensor")

		length = int(rollout["obs"].shape[0])
		if rollout["obs"].ndim < 2 or length <= 0:
			return False
		if rollout["elite_plans"].ndim != 4:
			raise ValueError("elite_plans must have shape (L, E, H, A)")
		if rollout["elite_mask"].ndim != 2:
			raise ValueError("elite_mask must have shape (L, E)")
		if rollout["pre_plan_mean"].ndim != 3:
			raise ValueError("pre_plan_mean must have shape (L, H, A)")
		if any(int(rollout[key].shape[0]) != length for key in self._REQUIRED_KEYS):
			raise ValueError("all post rollout tensors must share the same length")
		if tuple(rollout["elite_plans"].shape[:2]) != tuple(rollout["elite_mask"].shape):
			raise ValueError("elite_mask must align with elite_plans (L, E)")
		if tuple(rollout["elite_plans"].shape[2:]) != tuple(rollout["pre_plan_mean"].shape[1:]):
			raise ValueError("elite plans and pre_plan_mean must share (H, A)")
		if tuple(rollout["selected_plan"].shape[1:]) != tuple(rollout["pre_plan_mean"].shape[1:]):
			raise ValueError("selected_plan and pre_plan_mean must share (H, A)")
		if tuple(rollout["proposed_action"].shape) != tuple(rollout["executed_action"].shape):
			raise ValueError("proposed_action and executed_action must align")
		if tuple(rollout["proposed_action"].shape[1:]) != tuple(rollout["pre_plan_mean"].shape[2:]):
			raise ValueError("actions must share the planner action dimension")

		item = {
			key: value.detach().to("cpu").contiguous()
			for key, value in rollout.items()
			if torch.is_tensor(value)
		}
		item["elite_mask"] = item["elite_mask"].bool()
		item["collection_id"] = int(
			len(self._items) if collection_id is None else collection_id
		)
		item["model_update"] = int(
			item["collection_id"] if model_update is None else model_update
		)
		self._items.append(item)
		return True

	def sample(
		self,
		batch_size,
		min_len=1,
		min_items=1,
		device=None,
		generator=None,
		current_collection=None,
		current_update=None,
		max_age=None,
	):
		"""Sample and pad structured rollouts, returning ``None`` if unavailable."""
		eligible = self._eligible_items(
			min_len=min_len,
			current_collection=current_collection,
			current_update=current_update,
			max_age=max_age,
		)
		if len(eligible) < max(1, int(min_items)):
			return None

		n = min(max(0, int(batch_size)), len(eligible))
		if n <= 0:
			return None
		indices = torch.randperm(len(eligible), generator=generator)[:n].tolist()
		picked = [eligible[index] for index in indices]
		max_len, max_elites, _ = padded_batch_layout(
			[int(item["obs"].shape[0]) for item in picked],
			[int(item["elite_plans"].shape[1]) for item in picked],
		)

		obs = picked[0]["obs"].new_zeros((n, max_len, *picked[0]["obs"].shape[1:]))
		plans = picked[0]["elite_plans"].new_zeros(
			(n, max_len, max_elites, *picked[0]["elite_plans"].shape[2:])
		)
		pre_mean = picked[0]["pre_plan_mean"].new_zeros(
			(n, max_len, *picked[0]["pre_plan_mean"].shape[1:])
		)
		selected_plan = picked[0]["selected_plan"].new_zeros(
			(n, max_len, *picked[0]["selected_plan"].shape[1:])
		)
		proposed_action = picked[0]["proposed_action"].new_zeros(
			(n, max_len, *picked[0]["proposed_action"].shape[1:])
		)
		executed_action = picked[0]["executed_action"].new_zeros(
			(n, max_len, *picked[0]["executed_action"].shape[1:])
		)
		step_mask = torch.zeros(n, max_len, dtype=torch.bool)
		elite_mask = torch.zeros(n, max_len, max_elites, dtype=torch.bool)
		lengths = torch.zeros(n, dtype=torch.long)
		collection_ids = torch.zeros(n, dtype=torch.long)
		model_updates = torch.zeros(n, dtype=torch.long)
		values = None
		if any("elite_values" in item for item in picked):
			values = torch.full((n, max_len, max_elites), -torch.inf)

		for row, item in enumerate(picked):
			length = int(item["obs"].shape[0])
			elites = int(item["elite_plans"].shape[1])
			obs[row, :length].copy_(item["obs"])
			plans[row, :length, :elites].copy_(item["elite_plans"])
			pre_mean[row, :length].copy_(item["pre_plan_mean"])
			selected_plan[row, :length].copy_(item["selected_plan"])
			proposed_action[row, :length].copy_(item["proposed_action"])
			executed_action[row, :length].copy_(item["executed_action"])
			step_mask[row, :length] = True
			elite_mask[row, :length, :elites] = item["elite_mask"]
			lengths[row] = length
			collection_ids[row] = int(item["collection_id"])
			model_updates[row] = int(item["model_update"])
			if values is not None and "elite_values" in item:
				values[row, :length, :elites].copy_(item["elite_values"].float())

		batch = {
			"obs": obs,
			"step_mask": step_mask,
			"elite_plans": plans,
			"elite_mask": elite_mask,
			"pre_plan_mean": pre_mean,
			"selected_plan": selected_plan,
			"proposed_action": proposed_action,
			"executed_action": executed_action,
			"lengths": lengths,
			"collection_id": collection_ids,
			"model_update": model_updates,
		}
		if values is not None:
			batch["elite_values"] = values
		if device is not None:
			batch = {
				key: value.to(device, non_blocking=True)
				for key, value in batch.items()
			}
		return batch

	def eligible_count(
		self,
		min_len=1,
		current_collection=None,
		current_update=None,
		max_age=None,
	):
		"""Number of non-stale rollouts satisfying the requested length."""
		return len(
			self._eligible_items(
				min_len=min_len,
				current_collection=current_collection,
				current_update=current_update,
				max_age=max_age,
			)
		)

	def _eligible_items(
		self,
		*,
		min_len,
		current_collection,
		current_update,
		max_age,
	):
		minimum = max(1, int(min_len))
		eligible = []
		for item in self._items:
			if int(item["obs"].shape[0]) < minimum:
				continue
			if max_age is not None and (
				current_update is not None or current_collection is not None
			):
				if current_update is not None:
					age = int(current_update) - int(item["model_update"])
				else:
					age = int(current_collection) - int(item["collection_id"])
				if age > int(max_age):
					continue
			eligible.append(item)
		return eligible

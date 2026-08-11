"""Small structured replay buffer for real post-intervention rollouts.

Each item contains post observations produced by the current deployed policy
and, when available, diagnostics from the *same* deployed CEM calls.  The
diagnostics are stop-gradient candidate plans; the world model always re-scores
them at the optimizer update that consumes the batch.
"""

from collections import deque

import torch

class CausalPostBuffer:
	"""Fixed-capacity ring buffer of real post observations."""

	_REQUIRED_KEYS = ("obs",)
	_OPTIONAL_ALIGNED_KEYS = (
		"elite_plans",
		"elite_values",
		"selected_plan",
		"mean_plan",
		"pre_plan_mean",
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
			rollout: mapping containing ``obs (L, ...)``.
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
		if any(int(rollout[key].shape[0]) != length for key in self._REQUIRED_KEYS):
			raise ValueError("all post rollout tensors must share the same length")
		for key in self._OPTIONAL_ALIGNED_KEYS:
			if key not in rollout:
				continue
			if not torch.is_tensor(rollout[key]):
				raise TypeError(f"rollout[{key!r}] must be a tensor")
			if int(rollout[key].shape[0]) != length:
				raise ValueError(f"rollout[{key!r}] must align with post observations")

		item = {
			key: value.detach().to("cpu").contiguous()
			for key, value in rollout.items()
			if torch.is_tensor(value)
		}
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
		max_len = max(int(item["obs"].shape[0]) for item in picked)

		obs = picked[0]["obs"].new_zeros((n, max_len, *picked[0]["obs"].shape[1:]))
		step_mask = torch.zeros(n, max_len, dtype=torch.bool)
		lengths = torch.zeros(n, dtype=torch.long)
		collection_ids = torch.zeros(n, dtype=torch.long)
		model_updates = torch.zeros(n, dtype=torch.long)
		for row, item in enumerate(picked):
			length = int(item["obs"].shape[0])
			obs[row, :length].copy_(item["obs"])
			step_mask[row, :length] = True
			lengths[row] = length
			collection_ids[row] = int(item["collection_id"])
			model_updates[row] = int(item["model_update"])
		batch = {
			"obs": obs,
			"step_mask": step_mask,
			"lengths": lengths,
			"collection_id": collection_ids,
			"model_update": model_updates,
		}
		optional_keys = [
			key
			for key in self._OPTIONAL_ALIGNED_KEYS
			if all(key in item for item in picked)
		]
		for key in optional_keys:
			tail_shape = tuple(picked[0][key].shape[1:])
			if any(tuple(item[key].shape[1:]) != tail_shape for item in picked):
				raise ValueError(f"incompatible {key} shapes in post buffer")
			padded = picked[0][key].new_zeros((n, max_len, *tail_shape))
			for row, item in enumerate(picked):
				length = int(item[key].shape[0])
				padded[row, :length].copy_(item[key])
			batch[key] = padded
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

from jaxrl_m.dataset import Dataset
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import ml_collections

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    reachability_horizon: int = 25
    reachability_anchor_eps: float = 0.75
    reachability_far_eps: float = 5.0
    reachability_negative_horizon: int = 200
    reachability_hard_negative_prob: float = 0.75
    reachability_anchor_max_candidates: int = 64
    reachability_anchor_sample_attempts: int = 16

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
            'reachability_horizon': 25,
            'reachability_anchor_eps': 0.75,
            'reachability_far_eps': 5.0,
            'reachability_negative_horizon': 200,
            'reachability_hard_negative_prob': 0.75,
            'reachability_anchor_max_candidates': 64,
            'reachability_anchor_sample_attempts': 16,
        })

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)
        self._build_reachability_spatial_index()

    def _build_reachability_spatial_index(self):
        observations = self.dataset['observations']
        if isinstance(observations, FrozenDict) or not hasattr(observations, 'shape') or observations.shape[-1] < 2:
            self.reachability_obs_xy = None
            self.reachability_spatial_bins = None
            return

        self.reachability_obs_xy = np.asarray(observations[..., :2])
        eps = max(float(self.reachability_anchor_eps), 1e-6)
        cells = np.floor(self.reachability_obs_xy / eps).astype(np.int32)
        spatial_bins = {}
        for idx, cell in enumerate(cells):
            key = (int(cell[0]), int(cell[1]))
            spatial_bins.setdefault(key, []).append(idx)
        self.reachability_spatial_bins = {
            key: np.asarray(indices, dtype=np.int32)
            for key, indices in spatial_bins.items()
        }

    def _query_reachability_anchors(self, xy):
        if self.reachability_obs_xy is None or self.reachability_spatial_bins is None:
            return None

        eps = max(float(self.reachability_anchor_eps), 1e-6)
        cell = np.floor(xy / eps).astype(np.int32)
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (int(cell[0] + dx), int(cell[1] + dy))
                if key in self.reachability_spatial_bins:
                    candidates.append(self.reachability_spatial_bins[key])
        if not candidates:
            return None

        candidates = np.concatenate(candidates)
        distances = np.linalg.norm(self.reachability_obs_xy[candidates] - xy, axis=-1)
        anchors = candidates[distances < eps]
        if anchors.size == 0:
            return None

        max_candidates = int(self.reachability_anchor_max_candidates)
        if max_candidates > 0 and anchors.size > max_candidates:
            anchors = np.random.choice(anchors, size=max_candidates, replace=False)
        return anchors

    def _sample_reachability_anchor(self, xy, fallback_idx):
        if self.reachability_obs_xy is None or self.reachability_spatial_bins is None:
            return fallback_idx

        eps = max(float(self.reachability_anchor_eps), 1e-6)
        cell = np.floor(xy / eps).astype(np.int32)
        neighbor_bins = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (int(cell[0] + dx), int(cell[1] + dy))
                if key in self.reachability_spatial_bins:
                    neighbor_bins.append(self.reachability_spatial_bins[key])
        if not neighbor_bins:
            return fallback_idx

        for _ in range(int(self.reachability_anchor_sample_attempts)):
            bin_indices = neighbor_bins[np.random.randint(len(neighbor_bins))]
            candidate = bin_indices[np.random.randint(len(bin_indices))]
            if np.linalg.norm(self.reachability_obs_xy[candidate] - xy) < eps:
                return candidate
        return fallback_idx

    def _sample_future_index(self, anchor_idx, min_offset, max_offset):
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, anchor_idx)]
        start = anchor_idx + min_offset
        end = min(anchor_idx + max_offset, final_state_indx)
        if start > end:
            return None
        return np.random.randint(start, end + 1)

    def _sample_far_index(self, xy):
        if self.reachability_obs_xy is None:
            return np.random.randint(self.dataset.size)

        far_eps = float(self.reachability_far_eps)
        goal_indx = np.random.randint(self.dataset.size)
        for _ in range(16):
            if np.linalg.norm(self.reachability_obs_xy[goal_indx] - xy) > far_eps:
                return goal_indx
            goal_indx = np.random.randint(self.dataset.size)
        return goal_indx

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample_reachability_goals(self, indx):
        batch_size = len(indx)
        labels = (np.random.rand(batch_size) < 0.5).astype(np.float32)
        reach_goal_indx = np.empty(batch_size, dtype=np.int32)

        horizon = int(self.reachability_horizon)
        negative_horizon = max(int(self.reachability_negative_horizon), horizon + 1)

        for i, state_idx in enumerate(indx):
            if self.reachability_obs_xy is None:
                xy = None
                anchor_idx = state_idx
            else:
                xy = self.reachability_obs_xy[state_idx]
                anchor_idx = self._sample_reachability_anchor(xy, state_idx)

            goal_idx = None
            if labels[i] > 0.5:
                goal_idx = self._sample_future_index(anchor_idx, 1, horizon)
                if goal_idx is None:
                    goal_idx = self._sample_far_index(xy) if xy is not None else np.random.randint(self.dataset.size)
                    labels[i] = 0.0
            else:
                use_hard_negative = np.random.rand() < self.reachability_hard_negative_prob
                if use_hard_negative:
                    goal_idx = self._sample_future_index(anchor_idx, horizon + 1, negative_horizon)
                if goal_idx is None:
                    goal_idx = self._sample_far_index(xy) if xy is not None else np.random.randint(self.dataset.size)

            reach_goal_indx[i] = goal_idx
        return reach_goal_indx, labels

    def sample(self, batch_size: int, indx=None, sample_reachability=True):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
        if sample_reachability:
            reach_goal_indx, reach_labels = self.sample_reachability_goals(indx)
            batch['reachability_goals'] = jax.tree_map(lambda arr: arr[reach_goal_indx], self.dataset['observations'])
            batch['reachability_labels'] = reach_labels

        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    way_steps: int = None
    high_p_randomgoal: float = 0.

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': 0.0,
            'terminal': False,
            'reachability_horizon': 25,
            'reachability_anchor_eps': 0.75,
            'reachability_far_eps': 5.0,
            'reachability_negative_horizon': 200,
            'reachability_hard_negative_prob': 0.75,
            'reachability_anchor_max_candidates': 64,
            'reachability_anchor_sample_attempts': 16,
        })

    def sample(self, batch_size: int, indx=None, sample_reachability=True):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift

        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)

        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        way_indx = np.minimum(indx + self.way_steps, final_state_indx)
        batch['low_goals'] = jax.tree_map(lambda arr: arr[way_indx], self.dataset['observations'])

        distance = np.random.rand(batch_size)

        high_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        high_traj_target_indx = np.minimum(indx + self.way_steps, high_traj_goal_indx)

        high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        high_random_target_indx = np.minimum(indx + self.way_steps, final_state_indx)

        pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
        high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
        high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)

        batch['high_goals'] = jax.tree_map(lambda arr: arr[high_goal_idx], self.dataset['observations'])
        batch['high_targets'] = jax.tree_map(lambda arr: arr[high_target_idx], self.dataset['observations'])
        if sample_reachability:
            reach_goal_indx, reach_labels = self.sample_reachability_goals(indx)
            batch['reachability_goals'] = jax.tree_map(lambda arr: arr[reach_goal_indx], self.dataset['observations'])
            batch['reachability_labels'] = reach_labels

        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch

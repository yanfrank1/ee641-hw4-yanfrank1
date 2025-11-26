"""
Experience replay buffer for multi-agent DQN training.
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            state_A: Agent A's observation
            state_B: Agent B's observation
            action_A: Agent A's action
            action_B: Agent B's action
            comm_A: Communication from A to B
            comm_B: Communication from B to A
            reward: Shared reward
            next_state_A: Agent A's next observation
            next_state_B: Agent B's next observation
            done: Whether episode terminated
        """
        # TODO: Create transition tuple
        # TODO: Add to buffer (automatic removal of oldest if at capacity)
        transition = (
            state_A, state_B,
            action_A, action_B,
            comm_A, comm_B,
            reward,
            next_state_A, next_state_B,
            done
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions as separate arrays for each component
        """
        # TODO: Sample batch_size transitions randomly
        # TODO: Separate components into individual arrays
        # TODO: Convert to appropriate numpy arrays
        # TODO: Return tuple of arrays

        batch = random.sample(self.buffer, batch_size)
        states_A, states_B = [], []
        actions_A, actions_B = [], []
        comms_A, comms_B = [], []
        rewards = []
        next_states_A, next_states_B = [], []
        dones = []

        for t in batch:
            s_A, s_B, a_A, a_B, c_A, c_B, r, ns_A, ns_B, d = t
            states_A.append(s_A)
            states_B.append(s_B)
            actions_A.append(a_A)
            actions_B.append(a_B)
            comms_A.append(c_A)
            comms_B.append(c_B)
            rewards.append(r)
            next_states_A.append(ns_A)
            next_states_B.append(ns_B)
            dones.append(d)

        return (
            np.array(states_A, dtype=np.float32),
            np.array(states_B, dtype=np.float32),
            np.array(actions_A, dtype=np.int64),
            np.array(actions_B, dtype=np.int64),
            np.array(comms_A, dtype=np.float32),
            np.array(comms_B, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states_A, dtype=np.float32),
            np.array(next_states_B, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )


    def __len__(self) -> int:
        """
        Get current size of buffer.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for importance sampling.

    Samples transitions based on TD-error magnitude.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_steps: Steps to anneal beta to 1.0
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 1

        # TODO: Initialize data storage
        # TODO: Initialize priority tree (sum-tree or similar)
        # TODO: Set random seed if provided
        self.tree = SumTree(capacity)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store transition with maximum priority.

        New transitions get maximum priority to ensure they're sampled at least once.
        """
        # TODO: Store transition
        # TODO: Assign maximum priority to new transition
        max_priority = np.max(self.tree.tree[-self.capacity:])  
        if max_priority <= 0:
            max_priority = 1.0
        transition = (
            state_A, state_B,
            action_A, action_B,
            comm_A, comm_B,
            reward,
            next_state_A, next_state_B,
            done
        )
        self.tree.add(max_priority, transition)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with prioritization.

        Returns:
            transitions: Batch of transitions
            weights: Importance sampling weights
            indices: Indices for updating priorities
        """
        # TODO: Update beta based on schedule
        # TODO: Sample transitions based on priorities
        # TODO: Calculate importance sampling weights
        # TODO: Return transitions, weights, and indices
        self.beta = min(1.0, self.beta_start + (1 - self.beta_start) * (self.frame / self.beta_steps))
        self.frame += 1
        segment = self.tree.total() / batch_size
        samples = []
        indices = []
        priorities = []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                while data is None:
                    s = random.uniform(0, self.tree.total())
                    idx, p, data = self.tree.get(s)
            samples.append(data)
            indices.append(idx)
            priorities.append(p)

        priorities = np.array(priorities, dtype=np.float32)
        eps = 1e-6
        probs = np.maximum(priorities / (self.tree.total() + eps), eps)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= (weights.max() + eps)
        batch = list(zip(*samples))
        state_A = np.array(batch[0], dtype=np.float32)
        state_B = np.array(batch[1], dtype=np.float32)
        action_A = np.array(batch[2], dtype=np.int64)
        action_B = np.array(batch[3], dtype=np.int64)
        comm_A   = np.array(batch[4], dtype=np.float32)
        comm_B   = np.array(batch[5], dtype=np.float32)
        reward   = np.array(batch[6], dtype=np.float32)
        next_A   = np.array(batch[7], dtype=np.float32)
        next_B   = np.array(batch[8], dtype=np.float32)
        done     = np.array(batch[9], dtype=np.float32)
        
        out = (
            state_A, state_B,
            action_A, action_B,
            comm_A, comm_B,
            reward,
            next_A, next_B,
            done
        )
        return out, weights.astype(np.float32), indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD-errors)
        """
        # TODO: Update priorities for given indices
        # TODO: Apply alpha exponent for prioritization
        for idx, p in zip(indices, priorities):
            p = (np.abs(p) + 1e-6) ** self.alpha
            self.tree.update(idx, p)
    
    def __len__(self):
        return self.tree.size

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, value: float):
        parent = 0
        while parent < self.capacity - 1:
            left = 2 * parent + 1
            right = left + 1

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        leaf_idx = parent
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self) -> float:
        return self.tree[0]

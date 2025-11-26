"""
Stochastic gridworld environment for reinforcement learning.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class GridWorldEnv:
    """
    5x5 Stochastic GridWorld Environment.

    The agent navigates a grid with stochastic transitions:
    - 0.8 probability of moving in the intended direction
    - 0.1 probability of drifting left (perpendicular)
    - 0.1 probability of drifting right (perpendicular)

    Grid layout:
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (2, 2), (1, 3)
    - Penalties: (3, 1), (0, 3)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize gridworld environment.

        Args:
            seed: Random seed for reproducibility
        """
        self.grid_size = 5
        self.max_steps = 50

        # Define special cells
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

        # Rewards
        self.goal_reward = 10.0
        self.penalty_reward = -5.0
        self.step_cost = -0.1

        # Transition probabilities
        self.prob_intended = 0.8
        self.prob_drift = 0.1

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        if seed is not None:
            np.random.seed(seed)

        self.reset()

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            state: Initial state index
        """
        # TODO: Initialize agent position to start_pos
        # TODO: Reset step counter
        # TODO: Set done flag to False
        # TODO: Return state index (use _pos_to_state)
        self.agent_pos = self.start_pos
        self.steps = 0
        self.done = False
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index (0-3)

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode terminated
            info: Additional information
        """
        # TODO: Check if episode already done
        # TODO: Get next position based on stochastic transitions
        # TODO: Calculate reward (use _calculate_reward helper)
        # TODO: Update position and step count
        # TODO: Check termination conditions
        # TODO: Return (next_state, reward, done, info)
        if self.done:
            state = self._pos_to_state(self.agent_pos)
            return state, 0.0, True, {"msg": "Episode already terminated."}

        outcomes = self._get_next_positions(self.agent_pos, action)
        positions, probs = zip(*outcomes)
        idx = np.random.choice(len(positions), p=np.array(probs, dtype=float))
        next_pos = positions[idx]
        reward = self._calculate_reward(next_pos)
        self.agent_pos = next_pos
        self.steps += 1
        if self.agent_pos == self.goal_pos or self.steps >= self.max_steps:
            self.done = True

        next_state = self._pos_to_state(self.agent_pos)
        info = {
            "position": self.agent_pos,
            "steps": self.steps,
        }
        return next_state, reward, self.done, info

    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s'|s,a).

        Args:
            state: Current state index
            action: Action index

        Returns:
            Dictionary mapping next_state -> probability
        """
        # TODO: Convert state to position
        # TODO: For given action, compute all possible next positions
        #       considering stochastic transitions
        # TODO: Handle boundary and obstacle collisions
        # TODO: Return probability distribution over next states
        pos = self._state_to_pos(state)
        if self.is_terminal(state):
            return {state: 1.0}

        outcome_dict: Dict[Tuple[int, int], float] = {}
        for next_pos, prob in self._get_next_positions(pos, action):
            outcome_dict[next_pos] = outcome_dict.get(next_pos, 0.0) + prob

        trans_probs: Dict[int, float] = {}
        for p, prob in outcome_dict.items():
            s_next = self._pos_to_state(p)
            trans_probs[s_next] = trans_probs.get(s_next, 0.0) + prob

        return trans_probs

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition.

        Args:
            state: Current state index
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        # TODO: Convert next_state to position
        # TODO: Check if goal reached (+10)
        # TODO: Check if penalty cell (-5)
        # TODO: Otherwise return step cost (-0.1)
        next_pos = self._state_to_pos(next_state)
        return self._calculate_reward(next_pos)


    def is_terminal(self, state: int) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State index

        Returns:
            True if terminal state
        """
        # TODO: Convert state to position
        # TODO: Return True if position equals goal_pos
        pos = self._state_to_pos(state)
        return pos == self.goal_pos

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        Convert grid position to state index.

        Args:
            pos: (row, col) position

        Returns:
            State index (0-24)
        """
        # TODO: Convert 2D position to 1D state index
        # State = row * grid_size + col
        row, col = pos
        return row * self.grid_size + col

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position.

        Args:
            state: State index

        Returns:
            (row, col) position
        """
        # TODO: Convert 1D state index to 2D position
        # row = state // grid_size
        # col = state % grid_size
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        # TODO: Check if position is not an obstacle
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return False
        if pos in self.obstacles:
            return False
        return True

    def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get possible next positions and probabilities for stochastic transition.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            List of (next_position, probability) tuples
        """
        # TODO: Define action effects (deltas for UP, RIGHT, DOWN, LEFT)
        # TODO: Get intended direction and perpendicular directions
        # TODO: For each possible outcome (intended, drift left, drift right):
        #       - Calculate next position
        #       - If invalid, stay in current position
        #       - Add (position, probability) to list
        # TODO: Merge probabilities for same positions
        deltas = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        
        intended = action
        drift_left = (action - 1) % 4
        drift_right = (action + 1) % 4
        outcomes = [
            (intended, self.prob_intended),
            (drift_left, self.prob_drift),
            (drift_right, self.prob_drift),
        ]
        
        r, c = pos
        pos_prob: Dict[Tuple[int, int], float] = {}

        for act_idx, prob in outcomes:
            dr, dc = deltas[act_idx]
            candidate = (r + dr, c + dc)
            next_pos = candidate if self._is_valid_pos(candidate) else pos
            pos_prob[next_pos] = pos_prob.get(next_pos, 0.0) + prob

        return list(pos_prob.items())

    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """
        Calculate reward for entering a position.

        Args:
            pos: Position entered

        Returns:
            Reward value
        """
        # TODO: Check if position is goal (+10)
        # TODO: Check if position is penalty (-5)
        # TODO: Otherwise return step cost (-0.1)
        if pos == self.goal_pos:
            return self.goal_reward
        if pos in self.penalties:
            return self.penalty_reward
        return self.step_cost

    def render(self, value_function: Optional[np.ndarray] = None) -> None:
        """
        Render current state of environment.

        Args:
            value_function: Optional value function to display
        """
        # TODO: Create visual representation of grid
        # TODO: Mark current position, goal, obstacles, penalties
        # TODO: If value_function provided, show as heatmap
        grid = np.full((self.grid_size, self.grid_size), ' . ', dtype=object)
        
        for r, c in self.obstacles:
            grid[r, c] = ' X '
        for r, c in self.penalties:
            grid[r, c] = ' P '
        sr, sc = self.start_pos
        gr, gc = self.goal_pos
        grid[sr, sc] = ' S '
        grid[gr, gc] = ' G '
        
        ar, ac = self.agent_pos
        grid[ar, ac] = ' A '
        
        print("GridWorld:")
        for r in range(self.grid_size):
            print(''.join(grid[r, :]))

        if value_function is not None:
            vf_grid = value_function.reshape(self.grid_size, self.grid_size)
            with np.printoptions(precision=2, suppress=True):
                print(vf_grid)
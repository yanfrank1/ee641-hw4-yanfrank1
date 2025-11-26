"""
Multi-agent gridworld environment with partial observations and communication.
"""

import numpy as np
from typing import Tuple, Optional, List


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes a 3x3 local patch and exchanges communication signals.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Tuple defining grid dimensions (default 10x10)
            obs_window: Size of local observation window (must be odd, default 3)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        self.agent_positions = [None, None]
        self.comm_signals = [0.0, 0.0]
        self.step_count = 0

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target.

        Grid values:
        - 0: Free cell
        - 1: Obstacle
        - 2: Target
        """
        # TODO: Create empty grid of size grid_size
        # TODO: Randomly place up to 6 obstacles (avoiding corners)
        # TODO: Randomly place exactly 1 target cell
        # TODO: Store grid as self.grid
        h, w = self.grid_size
        grid = np.zeros((h, w), dtype=np.int32)
        target_row = np.random.randint(0, h)
        target_col = np.random.randint(0, w)
        grid[target_row, target_col] = 2
        self.target_pos = (target_row, target_col)
        
        corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        candidate_cells = [
            (x, y)
            for x in range(h)
            for y in range(w)
            if (x, y) not in corners and (x, y) != self.target_pos
        ]

        num_obstacles = np.random.randint(0, min(6, len(candidate_cells)) + 1)
        if num_obstacles > 0:
            obstacle_indices = np.random.choice(len(candidate_cells), size=num_obstacles, replace=False)
            for idx in obstacle_indices:
                x, y = candidate_cells[idx]
                grid[x, y] = 1

        self.grid = grid

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state.

        Returns:
            obs_A: Observation for Agent A (11-dimensional vector)
            obs_B: Observation for Agent B (11-dimensional vector)

        Observation format:
        - Elements 0-8: Flattened 3x3 grid patch (row-major order)
        - Element 9: Communication signal from other agent
        - Element 10: Normalized L2 distance between agents
        """
        # TODO: Reset step counter
        # TODO: Randomly place both agents on free cells (not obstacles or target)
        # TODO: Initialize communication signals to 0.0
        # TODO: Generate observations for both agents
        self.step_count = 0
        free_cells = self._find_free_cells()
        placements = np.random.choice(len(free_cells), size=2, replace=False)
        self.agent_positions[0] = free_cells[placements[0]]
        self.agent_positions[1] = free_cells[placements[1]]
        self.comm_signals = [0.0, 0.0]
        obs_A = self._get_observation(0).astype(np.float32)
        obs_B = self._get_observation(1).astype(np.float32)
        return obs_A, obs_B

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Execute one environment step.

        Args:
            action_A: Agent A's movement action (0:Up, 1:Down, 2:Left, 3:Right, 4:Stay)
            action_B: Agent B's movement action
            comm_A: Communication signal from Agent A to B
            comm_B: Communication signal from Agent B to A

        Returns:
            observations: Tuple of (obs_A, obs_B), each 11-dimensional
            reward: +10 if both agents at target, +2 if one agent at target, -0.1 per step
            done: True if both agents at target or max steps reached
        """
        # TODO: Update agent positions based on actions
        #       - Check boundaries and obstacles
        #       - Invalid moves result in no position change
        # TODO: Store new communication signals for next observation
        # TODO: Check reward condition (both agents at target)
        # TODO: Update step count and check termination
        # TODO: Generate new observations with updated comm signals
        pos_A = self.agent_positions[0]
        pos_B = self.agent_positions[1]
        new_pos_A = self._apply_action(pos_A, action_A)
        new_pos_B = self._apply_action(pos_B, action_B)
        self.agent_positions[0] = new_pos_A
        self.agent_positions[1] = new_pos_B
        self.comm_signals[0] = float(np.clip(comm_A, 0.0, 1.0))
        self.comm_signals[1] = float(np.clip(comm_B, 0.0, 1.0))
        target_x, target_y = self.target_pos
        at_target_A = (new_pos_A[0] == target_x and new_pos_A[1] == target_y)
        at_target_B = (new_pos_B[0] == target_x and new_pos_B[1] == target_y)
        
        if at_target_A and at_target_B:
            reward = 10.0
        elif at_target_A or at_target_B:
            reward = 2.0
        else:
            reward = -0.1
        
        self.step_count += 1
        done = (at_target_A or at_target_B) or (self.step_count >= self.max_steps)
        obs_A = self._get_observation(0).astype(np.float32)
        obs_B = self._get_observation(1).astype(np.float32)
        return (obs_A, obs_B), float(reward), bool(done)

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Extract local observation for an agent.

        Args:
            agent_idx: Agent index (0 for A, 1 for B)

        Returns:
            observation: 10-dimensional vector
        """
        # TODO: Get agent position
        # TODO: Extract 3x3 patch centered on agent
        #       - Cells outside grid should be -1
        #       - Use grid values (0: free, 1: obstacle, 2: target)
        # TODO: Flatten patch to 9 elements
        # TODO: Append communication signal from other agent
        # TODO: Return 10-dimensional observation
        h, w = self.grid_size
        x, y = self.agent_positions[agent_idx]
        patch = np.full((3, 3), -1, dtype=np.int32)
        for x2 in range(-1, 2):
            for y2 in range(-1, 2):
                new_x = x + x2
                new_y = y + y2
                if 0 <= new_x < h and 0 <= new_y < w:
                    patch[x2 + 1, y2 + 1] = self.grid[new_x, new_y]

        patch_flat = patch.flatten().astype(np.float32)
        comm_in = float(self.comm_signals[1 - agent_idx])
        (x1, y1) = self.agent_positions[0]
        (x2, y2) = self.agent_positions[1]
        num = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        h, w = self.grid_size
        denom = np.sqrt(h ** 2 + w ** 2)
        dist = np.float32(num / denom)
        obs = np.concatenate([patch_flat, np.array([comm_in, dist], dtype=np.float32)]).astype(np.float32)
        return obs

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        # TODO: Check if position is not an obstacle (grid value != 1)
        h, w = self.grid_size
        if not (0 <= pos[0] < h and 0 <= pos[1] < w):
            return False
        if self.grid[pos[0], pos[1]] == 1:
            return False
        return True

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply movement action to position.

        Args:
            pos: Current position (row, col)
            action: Movement action (0-4)

        Returns:
            new_pos: Updated position (stays same if invalid)
        """
        # TODO: Map action to position delta
        #       0: Up (-1, 0)
        #       1: Down (+1, 0)
        #       2: Left (0, -1)
        #       3: Right (0, +1)
        #       4: Stay (0, 0)
        # TODO: Calculate new position
        # TODO: Return new position if valid, else return original position
        x, y = pos
        if action == 0:
            x2, y2 = -1, 0
        elif action == 1:
            x2, y2 = 1, 0
        elif action == 2:
            x2, y2 = 0, -1
        elif action == 3:
            x2, y2 = 0, 1
        elif action == 4:
            x2, y2 = 0, 0

        new_pos = (x + x2, y + y2)
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return pos

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """
        Find all free cells in the grid.

        Returns:
            List of (row, col) positions that are free
        """
        # TODO: Iterate through grid
        # TODO: Collect positions where grid value is 0 (free)
        # TODO: Return list of free positions
        h, w = self.grid_size
        free_positions: List[Tuple[int, int]] = []
        for x in range(h):
            for y in range(w):
                if self.grid[x, y] == 0:
                    free_positions.append((x, y))
        return free_positions

    def render(self) -> None:
        """
        Render current environment state.
        """
        # TODO: Create visual representation of grid
        # TODO: Show agent positions (A, B)
        # TODO: Show target (T)
        # TODO: Show obstacles (X)
        # TODO: Display current communication values

        h, w = self.grid_size
        char_grid = np.full((h, w), ".", dtype=object)

        for x in range(h):
            for y in range(w):
                if self.grid[x, y] == 1:
                    char_grid[x, y] = "X"
                elif self.grid[x, y] == 2:
                    char_grid[x, y] = "T"

        if self.agent_positions[0] is not None:
            x1, y1 = self.agent_positions[0]
            char_grid[x1, y1] = "A"
        if self.agent_positions[1] is not None:
            x2, y2 = self.agent_positions[1]
            if char_grid[x2, y2] == "A":
                char_grid[x2, y2] = "Z"
            else:
                char_grid[x2, y2] = "B"

        print(f"Step: {self.step_count}")
        print("Comm signals: A->B={:.3f}, B->A={:.3f}".format(self.comm_signals[0], self.comm_signals[1]))
        for x in range(h):
            row_str = " ".join(char_grid[x, :])
            print(row_str)
        print()
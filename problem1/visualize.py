"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os


class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size

        # Define special positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

    def plot_value_function(self, values: np.ndarray, title: str = "Value Function") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
        """
        # TODO: Reshape values to 2D grid
        # TODO: Create heatmap with appropriate colormap
        # TODO: Mark special cells (start, goal, obstacles, penalties)
        # TODO: Add colorbar and labels
        # TODO: Save figure to results/visualizations/
        V = values.reshape(self.grid_size, self.grid_size)
        fig, ax = plt.subplots()
        im = ax.imshow(V, cmap="coolwarm", origin="lower")
        ax.invert_yaxis()
        
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, color="black", alpha=0.5))
        for x, y in self.penalties:
            ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, color="red", alpha=0.5))
        sx, sy = self.start_pos
        ax.add_patch(plt.Rectangle((sy-0.5, sx-0.5), 1, 1, color="blue", alpha=0.5))
        gx, gy = self.goal_pos
        ax.add_patch(plt.Rectangle((gy-0.5, gx-0.5), 1, 1, color="green", alpha=0.5))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                ax.text(y, x, f"{V[x, y]:.2f}", ha="center", va="center")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("Value")
        save_path = os.path.join("results/visualizations/", f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
        """
        # TODO: Create grid plot
        # TODO: For each state:
        #       - Draw arrow indicating action direction
        #       - Handle special cells appropriately
        # TODO: Mark start, goal, obstacles, penalties
        # TODO: Save figure to results/visualizations/
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.invert_yaxis()
        for x in range(self.grid_size + 1):
            ax.axvline(x, color="gray", linewidth=1)
            ax.axhline(x, color="gray", linewidth=1)

        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color="black", alpha=0.5))
        for x, y in self.penalties:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color="red", alpha=0.5))
        sx, sy = self.start_pos
        ax.add_patch(plt.Rectangle((sy, sx), 1, 1, color="blue", alpha=0.5))
        gx, gy = self.goal_pos
        ax.add_patch(plt.Rectangle((gy, gx), 1, 1, color="green", alpha=0.5))

        arrows = {
            0: (0, -0.3),
            1: (0.3, 0),
            2: (0, 0.3),
            3: (-0.3, 0),
        }

        for s in range(self.grid_size ** 2):
            x = s // self.grid_size
            y = s % self.grid_size
            if (x, y) in self.obstacles or (x, y) in self.penalties:
                continue
            if (x, y) == self.goal_pos:
                continue
            a = policy[s]
            dx, dy = arrows[a]
            ax.arrow(y + 0.5, x + 0.5, dx, dy, head_width=0.15, head_length=0.15, color="k")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        save_path = os.path.join("results/visualizations/", f"{title.replace(' ','_').lower()}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
        """
        # TODO: Create subplot for each action
        # TODO: For each action:
        #       - Show Q-values as heatmap
        #       - Mark special cells
        # TODO: Add overall title and save
        actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        fig, axes = plt.subplots(2, 2)
        vmin = q_values.min()
        vmax = q_values.max()

        for a in range(4):
            ax = axes[a // 2, a % 2]
            Q_a = q_values[:, a].reshape(self.grid_size, self.grid_size)
            im = ax.imshow(Q_a, cmap="coolwarm", origin="lower", vmin=vmin, vmax=vmax)
            ax.invert_yaxis()

            for x, y in self.obstacles:
                ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, color="black" , alpha=0.5))
            for x, y in self.penalties:
                ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, color="red", alpha=0.5))
            sx, sy = self.start_pos
            ax.add_patch(plt.Rectangle((sy-0.5, sx-0.5), 1, 1, color="blue", alpha=0.5))
            gx, gy = self.goal_pos
            ax.add_patch(plt.Rectangle((gy-0.5, gx-0.5), 1, 1, color="green", alpha=0.5))

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    ax.text(y, x, f"{Q_a[x,y]:.2f}", ha="center", va="center")

            ax.set_title(f"{title} – {actions[a]}")
            ax.set_xticks([])
            ax.set_yticks([])
            
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Q-value")
        save_path = os.path.join("results/visualizations/", f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

    def plot_convergence(self, vi_history: list, qi_history: list) -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history
            qi_history: Q-iteration convergence history
        """
        # TODO: Plot Bellman error vs iteration for both algorithms
        # TODO: Use log scale for y-axis
        # TODO: Add legend and labels
        # TODO: Save figure
        plt.figure()
        plt.plot(vi_history, label="Value Iteration")
        plt.plot(qi_history, label="Q-Iteration")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Bellman Error (log scale)")
        plt.title("Convergence Curve")
        plt.legend()
        save_path = os.path.join("results/visualizations/", "convergence_curve.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    
    def _plot_policy_on_ax(self, ax, policy, title):
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.invert_yaxis()
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color="black", alpha=0.5))
        for x, y in self.penalties:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color="red", alpha=0.5))
        sx, sy = self.start_pos
        ax.add_patch(plt.Rectangle((sy, sx), 1, 1, color="blue", alpha=0.5))
        gx, gy = self.goal_pos
        ax.add_patch(plt.Rectangle((gy, gx), 1, 1, color="green", alpha=0.5))

        arrows = {
            0: (0, -0.3),
            1: (0.3, 0),
            2: (0, 0.3),
            3: (-0.3, 0),
        }

        for s in range(self.grid_size ** 2):
            x = s // self.grid_size
            y = s % self.grid_size
            if (x, y) in self.obstacles or (x, y) in self.penalties:
                continue
            if (x, y) == self.goal_pos:
                continue
            a = policy[s]
            dx, dy = arrows[a]
            ax.arrow(y + 0.5, x + 0.5, dx, dy, head_width=0.15, color="k")

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        
    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
        """
        Create comparison figure showing both algorithms' results.

        Args:
            vi_values: Value function from Value Iteration
            qi_values: Value function from Q-Iteration
            vi_policy: Policy from Value Iteration
            qi_policy: Policy from Q-Iteration
        """
        # TODO: Create 2x2 subplot
        #       - Top left: VI value function
        #       - Top right: QI value function
        #       - Bottom left: VI policy
        #       - Bottom right: QI policy
        # TODO: Highlight any differences
        # TODO: Save comprehensive comparison figure
        fig, axes = plt.subplots(2, 2)
        V_vi = vi_values.reshape(self.grid_size, self.grid_size)
        V_qi = qi_values.reshape(self.grid_size, self.grid_size)
        ax = axes[0, 0]
        im = ax.imshow(V_vi, cmap="coolwarm", origin="lower")
        ax.invert_yaxis()
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color="black", alpha=0.5))
        for x, y in self.penalties:
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color="red", alpha=0.5))
        sx, sy = self.start_pos
        ax.add_patch(plt.Rectangle((sx-0.5, sy-0.5), 1, 1, color="blue", alpha=0.5))
        gx, gy = self.goal_pos
        ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5), 1, 1, color="green", alpha=0.5))
        ax.set_title("Value Iteration – Value Function")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax)

        ax = axes[0, 1]
        im = ax.imshow(V_qi, cmap="coolwarm", origin="upper")
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color="black", alpha=0.5))
        for x, y in self.penalties:
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color="red", alpha=0.5))
        sx, sy = self.start_pos
        ax.add_patch(plt.Rectangle((sx-0.5, sy-0.5), 1, 1, color="blue", alpha=0.5))
        gx, gy = self.goal_pos
        ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5), 1, 1, color="green", alpha=0.5))
        ax.set_title("Q-Iteration – Value Function")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax)

        self._plot_policy_on_ax(axes[1, 0], vi_policy, "Value Iteration – Policy")
        self._plot_policy_on_ax(axes[1, 1], qi_policy, "Q-Iteration – Policy")

        save_path = os.path.join("results/visualizations/", "comparison.png")
        plt.savefig(save_path, dpi=200)
        plt.close()


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    # TODO: Load saved value functions and policies
    # TODO: Create visualizer instance
    # TODO: Generate all visualization plots
    # TODO: Print summary statistics
    vi_values = np.load(os.path.join("results/", "vi_values.npy"))
    vi_policy = np.load(os.path.join("results/", "vi_policy.npy"))
    qi_values = np.load(os.path.join("results/", "qi_values.npy"))
    qi_policy = np.load(os.path.join("results/", "qi_policy.npy"))
    vi_history = np.load(os.path.join("results/", "vi_bellman_history.npy"))
    qi_history = np.load(os.path.join("results/", "qi_bellman_history.npy"))
    
    viz = GridWorldVisualizer(grid_size=5)
    viz.plot_value_function(vi_values, title="VI Value Function")
    viz.plot_value_function(qi_values, title="QI Value Function")
    viz.plot_policy(vi_policy, title="VI Policy")
    viz.plot_policy(qi_policy, title="QI Policy")
    viz.plot_q_function(
        q_values=np.load(os.path.join("results/", "qi_qvalues.npy"))
        if os.path.exists(os.path.join("results/", "qi_qvalues.npy"))
        else None,
        title="QI Q-Function"
    ) if os.path.exists(os.path.join("results/", "qi_qvalues.npy")) else None
    viz.plot_convergence(vi_history, qi_history)
    viz.create_comparison_figure(
        vi_values=vi_values,
        qi_values=qi_values,
        vi_policy=vi_policy,
        qi_policy=qi_policy
    )
    
    print("📈 Summary statistics:")
    print(f"  • VI: min={vi_values.min():.3f}, max={vi_values.max():.3f}")
    print(f"  • QI: min={qi_values.min():.3f}, max={qi_values.max():.3f}")
    policy_diff = np.sum(vi_policy != qi_policy)
    print(f"  • Policies differ in {policy_diff} of {len(vi_policy)} states")

if __name__ == '__main__':
    visualize_results()
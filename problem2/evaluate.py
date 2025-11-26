"""
Evaluation script for trained multi-agent models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from multi_agent_env import MultiAgentEnv
from models import DuelingDQN
from train import apply_observation_mask
import argparse

class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module, args):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        self.args = args
        # Use CPU for small networks
        self.device = torch.device("cpu")

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        self.model_A.eval()
        self.model_B.eval()
    
    def select_action(self, state: np.ndarray, network: nn.Module) -> Tuple[int, float]:
        masked_state = apply_observation_mask(state, self.args.mode)
        state_tensor = torch.tensor(masked_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values, comm_tensor = network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            comm_signal = float(comm_tensor.squeeze().cpu().numpy())

        comm_signal = float(np.clip(comm_signal, 0.0, 1.0))
        return (action, comm_signal)
        
    def run_episode(self, render: bool = False) -> Tuple[float, bool, Dict]:
        """
        Run single evaluation episode.

        Args:
            render: Whether to render environment

        Returns:
            reward: Episode reward
            success: Whether target was reached
            info: Episode statistics
        """
        # TODO: Reset environment
        # TODO: Initialize episode tracking
        # TODO: Run episode with greedy policy
        # TODO: Track communication patterns
        # TODO: Return results and statistics
        obs_A, obs_B = self.env.reset()
        total_reward = 0.0
        last_reward = 0.0
        done = False
        success = False
        step = 0
        target_x, target_y = self.env.target_pos
        H, W = self.env.grid_size
        max_dist = np.sqrt(H ** 2 + W ** 2)

        positions_A: List[Tuple[int, int]] = [self.env.agent_positions[0]]
        positions_B: List[Tuple[int, int]] = [self.env.agent_positions[1]]
        comm_A_list: List[float] = []
        comm_B_list: List[float] = []
        on_target_A: List[bool] = []
        on_target_B: List[bool] = []
        distances: List[float] = []

        while not done:
            if render:
                self.env.render()

            action_A, comm_A = self.select_action(obs_A, self.model_A)
            action_B, comm_B = self.select_action(obs_B, self.model_B)
            (next_A, next_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
            total_reward += reward
            last_reward = reward
            step += 1

            comm_A_list.append(comm_A)
            comm_B_list.append(comm_B)
            pos_A = self.env.agent_positions[0]
            pos_B = self.env.agent_positions[1]
            positions_A.append(pos_A)
            positions_B.append(pos_B)
            on_A = (pos_A[0] == target_x and pos_A[1] == target_y)
            on_B = (pos_B[0] == target_x and pos_B[1] == target_y)
            on_target_A.append(on_A)
            on_target_B.append(on_B)
            dist = np.sqrt((pos_A[0] - pos_B[0]) ** 2 + (pos_A[1] - pos_B[1]) ** 2) / max_dist
            distances.append(float(dist))
            obs_A, obs_B = next_A, next_B

        if done and last_reward == 10.0:
            success = True

        info = {
            "steps": step,
            "reward": float(total_reward),
            "success": success,
            "positions_A": positions_A,
            "positions_B": positions_B,
            "comm_A": comm_A_list,
            "comm_B": comm_B_list,
            "A_on_target": on_target_A,
            "B_on_target": on_target_B,
            "distances": distances,
            "target_pos": (target_x, target_y),
        }

        return total_reward, success, info

    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate overall performance statistics.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Statistics dictionary
        """
        # TODO: Run multiple episodes
        # TODO: Compute success rate
        # TODO: Analyze path lengths
        # TODO: Measure coordination efficiency
        # TODO: Return comprehensive statistics
        rewards = []
        successes = []
        steps_list = []
        total_one = []
        total_both = []
        coordination_delays = []

        for _ in range(num_episodes):
            ep_reward, success, info = self.run_episode(render=False)
            rewards.append(ep_reward)
            successes.append(1.0 if success else 0.0)
            steps_list.append(info["steps"])
            A_on = np.array(info["A_on_target"], dtype=bool)
            B_on = np.array(info["B_on_target"], dtype=bool)
            target_one = np.where(A_on | B_on)[0]
            if len(target_one) > 0:
                t_one = int(target_one[0])
                total_one.append(t_one)

            target_both = np.where(A_on & B_on)[0]
            if len(target_both) > 0:
                t_both = int(target_both[0])
                total_both.append(t_both)
                if len(target_one) > 0:
                    coordination_delays.append(t_both - t_one)

        stats: Dict[str, float] = {
            "num_episodes": num_episodes,
            "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            "success_rate": float(np.mean(successes)) if len(successes) > 0 else 0.0,
            "mean_steps": float(np.mean(steps_list)) if len(steps_list) > 0 else 0.0,
        }

        success_steps = [s for s, succ in zip(steps_list, successes) if succ > 0.5]
        if success_steps:
            stats["mean_steps_success"] = float(np.mean(success_steps))
        else:
            stats["mean_steps_success"] = None

        if total_one:
            stats["mean_one_success"] = float(np.mean(total_one))
        else:
            stats["mean_one_success"] = None

        if total_both:
            stats["mean_both_success"] = float(np.mean(total_both))
        else:
            stats["mean_both_success"] = None

        if coordination_delays:
            stats["coordination_delay_mean"] = float(np.mean(coordination_delays))
            stats["coordination_delay_std"] = float(np.std(coordination_delays))
        else:
            stats["coordination_delay_mean"] = None
            stats["coordination_delay_std"] = None

        return stats

    def analyze_communication(self, num_episodes: int = 20) -> Dict:
        """
        Analyze emergent communication protocols.

        Returns:
            Communication analysis results
        """
        # TODO: Track communication signals over episodes
        # TODO: Analyze signal patterns (magnitude, variance, correlation)
        # TODO: Identify communication strategies
        # TODO: Return analysis results
        all_comm_A = []
        all_comm_B = []
        all_distances = []
        for _ in range(num_episodes):
            _, _, info = self.run_episode(render=False)
            all_comm_A.extend(info["comm_A"])
            all_comm_B.extend(info["comm_B"])
            all_distances.extend(info["distances"])

        all_comm_A = np.array(all_comm_A, dtype=np.float32)
        all_comm_B = np.array(all_comm_B, dtype=np.float32)
        all_distances = np.array(all_distances, dtype=np.float32)

        results: Dict[str, float] = {
            "num_episodes": num_episodes,
            "num_samples": int(len(all_comm_A)),
        }

        results.update({
            "mean_comm_A": float(np.mean(all_comm_A)),
            "std_comm_A": float(np.std(all_comm_A)),
            "min_comm_A": float(np.min(all_comm_A)),
            "max_comm_A": float(np.max(all_comm_A)),
            "corr_comm_A": float(np.corrcoef(all_comm_A, all_distances)[0, 1]),
            "mean_comm_B": float(np.mean(all_comm_B)),
            "std_comm_B": float(np.std(all_comm_B)),
            "min_comm_B": float(np.min(all_comm_B)),
            "max_comm_B": float(np.max(all_comm_B)),
            "corr_comm_B": float(np.corrcoef(all_comm_B, all_distances)[0, 1])
        })

        return results

    def visualize_trajectory(self, save_path: str = 'results/trajectory.png') -> None:
        """
        Visualize agent trajectories in an episode.

        Args:
            save_path: Path to save visualization
        """
        # TODO: Run episode while tracking positions
        # TODO: Create grid visualization
        # TODO: Plot agent paths
        # TODO: Mark key events (near target, coordination points)
        # TODO: Save figure
        ep_reward, success, info = self.run_episode(render=False)
        positions_A = np.array(info["positions_A"])
        positions_B = np.array(info["positions_B"])
        target_x, target_y = info["target_pos"]
        grid = self.env.grid
        H, W = self.env.grid_size
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots()
        ax.imshow(grid, origin="upper")
        ax.plot(positions_A[:, 1], positions_A[:, 0], marker='o', linestyle='-', label='Agent A')
        ax.plot(positions_B[:, 1], positions_B[:, 0], marker='s', linestyle='-', label='Agent B')
        ax.scatter(positions_A[0, 1], positions_A[0, 0], marker='o', label='A start')
        ax.scatter(positions_B[0, 1], positions_B[0, 0], marker='s', label='B start')
        ax.scatter(target_x, target_y, marker='*', label='Target')
        ax.set_title(f"Trajectory (reward={ep_reward:.2f}, success={success})")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def plot_communication_heatmap(self, save_path: str = 'results/comm_heatmap.png') -> None:
        """
        Create heatmap of communication signals across grid positions.

        Args:
            save_path: Path to save figure
        """
        # TODO: Sample communication signals at each grid position
        # TODO: Create heatmaps for both agents
        # TODO: Show correlation with distance to target
        # TODO: Save visualization
        H, W = self.env.grid_size
        comm_sum_A = np.zeros((H, W), dtype=np.float32)
        comm_sum_B = np.zeros((H, W), dtype=np.float32)
        count_A = np.zeros((H, W), dtype=np.int32)
        count_B = np.zeros((H, W), dtype=np.int32)

        for _ in range(50):
            _, _, info = self.run_episode(render=False)
            positions_A = info["positions_A"]
            positions_B = info["positions_B"]
            comm_A = info["comm_A"]
            comm_B = info["comm_B"]

            for t, yA in enumerate(comm_A):
                xA, yA_pos = positions_A[t]
                comm_sum_A[xA, yA_pos] += yA
                count_A[xA, yA_pos] += 1

            for t, yB in enumerate(comm_B):
                xB, yB_pos = positions_B[t]
                comm_sum_B[xB, yB_pos] += yB
                count_B[xB, yB_pos] += 1

        mean_comm_A = np.zeros((H, W), dtype=np.float32)
        mean_comm_B = np.zeros((H, W), dtype=np.float32)
        mask_A = count_A > 0
        mask_B = count_B > 0
        mean_comm_A[mask_A] = comm_sum_A[mask_A] / count_A[mask_A]
        mean_comm_B[mask_B] = comm_sum_B[mask_B] / count_B[mask_B]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2)
        im1 = axes[0].imshow(mean_comm_A, origin="upper")
        axes[0].set_title("Mean communication (Agent A)")
        axes[0].set_xticks(range(W))
        axes[0].set_yticks(range(H))
        im2 = axes[1].imshow(mean_comm_B, origin="upper")
        axes[1].set_title("Mean communication (Agent B)")
        axes[1].set_xticks(range(W))
        axes[1].set_yticks(range(H))
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def test_generalization(self, num_configs: int = 10) -> Dict:
        """
        Test generalization to new environment configurations.

        Args:
            num_configs: Number of test configurations

        Returns:
            Generalization performance statistics
        """
        # TODO: Generate new obstacle configurations
        # TODO: Test performance on each configuration
        # TODO: Compare to training performance
        # TODO: Return generalization metrics
        base_stats = self.evaluate_performance()
        config_stats_list = []
        H, W = self.env.grid_size
        base_max_steps = self.env.max_steps
        
        for i in range(num_configs):
            seed = np.random.randint(0, 10_000_000)
            new_env = MultiAgentEnv(grid_size=(H, W), max_steps=base_max_steps, seed=seed)
            evaluator = MultiAgentEvaluator(new_env, self.model_A, self.model_B, self.args)
            stats = evaluator.evaluate_performance()
            stats["config_id"] = i
            config_stats_list.append(stats)

        gen_success_rates = [cs["success_rate"] for cs in config_stats_list]
        gen_mean_rewards = [cs["mean_reward"] for cs in config_stats_list]
        gen_summary = {
            "num_configs": num_configs,
            "base_env_performance": base_stats,
            "configs": config_stats_list,
            "mean_generalization_success_rate": float(np.mean(gen_success_rates)) if gen_success_rates else None,
            "mean_generalization_reward": float(np.mean(gen_mean_rewards)) if gen_mean_rewards else None,
        }

        return gen_summary

def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained agent models from checkpoint.

    Args:
        checkpoint_dir: Directory containing saved models

    Returns:
        model_A: Agent A's trained model
        model_B: Agent B's trained model
    """
    # TODO: Load model architectures
    # TODO: Load trained weights
    # TODO: Return initialized models
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_ep") and f.endswith(".pth")]
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.replace("checkpoint_ep", "").replace(".pth", "")))
    latest_ckpt = ckpts_sorted[-1]
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_A = DuelingDQN(input_dim=11, hidden_dim=64, num_actions=5)
    model_B = DuelingDQN(input_dim=11, hidden_dim=64, num_actions=5)
    model_A.load_state_dict(checkpoint["network_A"])
    model_B.load_state_dict(checkpoint["network_B"])
    model_A.eval()
    model_B.eval()

    return model_A, model_B


def create_evaluation_report(results: Dict, save_path: str = 'results/evaluation_report.json') -> None:
    """
    Create comprehensive evaluation report.

    Args:
        results: Evaluation results
        save_path: Path to save report
    """
    # TODO: Format results
    # TODO: Add summary statistics
    # TODO: Save as JSON report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    def _sanitize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    sanitized = _sanitize(results)
    with open(save_path, "w") as f:
        json.dump(sanitized, f, indent=2)

    print(f"Saved evaluation report to {save_path}")

def plot_training_curve(log_path: str, save_path: str):
    with open(log_path, "r") as f:
        log = json.load(f)

    rewards = log["rewards"]
    window = 50
    average = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.figure()
    plt.plot(rewards, linewidth=1, label="Raw Reward")
    plt.plot(np.arange(len(average)), average, linewidth=1, label=f"Smoothed (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve (Rewards)")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Run full evaluation suite on trained models.
    """
    # TODO: Load trained models
    # TODO: Create environment
    # TODO: Initialize evaluator
    # TODO: Run performance evaluation
    # TODO: Analyze communication
    # TODO: Test generalization
    # TODO: Create visualizations
    # TODO: Generate report
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'])
    args = parser.parse_args()
    
    checkpoint_dir = os.path.join("results", args.mode)
    model_A, model_B = load_trained_models(checkpoint_dir)
    env = MultiAgentEnv(grid_size=(10, 10), max_steps=50, seed=123)
    evaluator = MultiAgentEvaluator(env, model_A, model_B, args)
    performance_stats = evaluator.evaluate_performance(num_episodes=100)
    print("Performance stats:")
    print(performance_stats)
    comm_stats = evaluator.analyze_communication(num_episodes=20)
    print("Communication stats:")
    print(comm_stats)
    gen_stats = evaluator.test_generalization(num_configs=10)
    print("Generalization stats:")
    print("Base env:", gen_stats["base_env_performance"])
    print("Mean generalization success rate:", gen_stats["mean_generalization_success_rate"])
    evaluator.visualize_trajectory(save_path=os.path.join(checkpoint_dir, "trajectory.png"))
    evaluator.plot_communication_heatmap(save_path=os.path.join(checkpoint_dir,"comm_heatmap.png"))
    results = {
        "performance": performance_stats,
        "communication": comm_stats,
        "generalization": gen_stats,
    }
    training_log_path = os.path.join(checkpoint_dir, "training_log.json")
    plot_training_curve(training_log_path, save_path=os.path.join(checkpoint_dir, "training_curve.png"))
    create_evaluation_report(results, save_path=os.path.join(checkpoint_dir,"evaluation_report.json"))


if __name__ == '__main__':
    main()